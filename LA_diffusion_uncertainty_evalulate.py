import os
import sys
import json
import time
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
from datetime import datetime

from monai.data import CacheDataset, ThreadDataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, compute_hausdorff_distance, compute_average_surface_distance
from monai.transforms import (
    AsDiscrete, Compose, EnsureChannelFirstd, EnsureTyped,
    LoadImaged, NormalizeIntensityd, Orientationd, Lambdad
)
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, UniPCMultistepScheduler

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

PATCH_SIZE = (288, 192, 32)
root_dir    = '/data/donnees/student1/jzhang/'
data_dir   = os.path.join(root_dir, "Lascarqs_stage1/train_data_cropped_130_384_256_40/") # LA path
json_path  = data_dir + "datalist_5fold.json"
model_dir   = os.path.join(root_dir, "LA_diff_results_2026_05_07_15_10/")
timestamp   = datetime.now().strftime("%Y_%m_%d_%H_%M")
results_dir = os.path.join(root_dir, f"LA_diff_uncertainty_results_{timestamp}")

# =====================================================================
# Scheduler configurations — shared base args
# All schedulers use the same noise schedule as training
# =====================================================================
SCHEDULER_BASE_KWARGS = dict(
    num_train_timesteps=1000,
    beta_start=0.0005,
    beta_end=0.0195,
    beta_schedule="linear",
    prediction_type="sample",
)

SCHEDULERS = {
    "DDIM": lambda: DDIMScheduler(**SCHEDULER_BASE_KWARGS, clip_sample=True,),
    "DPMSolverMultistep": lambda: DPMSolverMultistepScheduler(**SCHEDULER_BASE_KWARGS,),
    "EulerDiscrete": lambda: EulerDiscreteScheduler(**SCHEDULER_BASE_KWARGS,),
    "UniPCMultistep": lambda: UniPCMultistepScheduler(**SCHEDULER_BASE_KWARGS,),
}

# Inference steps to benchmark
INFERENCE_STEPS_LIST = [20, 50]

# Number of stochastic samples for uncertainty estimation.
# Use more samples than speed benchmarking; 10-30 is a practical range.
N_SAMPLES = 20

# Save each stochastic binary sample as NIfTI. This can use a lot of disk space.
SAVE_INDIVIDUAL_SAMPLES = False

UNCERTAINTY_EPS = 1e-6


# =====================================================================
# Dynamic Conditional Encoder (DCE)
# =====================================================================
class DynamicCondEncoder(nn.Module):

    def __init__(self, in_channels: int = 1, base_channels: int = 24,
                 channel_mult: tuple = (1, 2, 4, 8), time_emb_dim: int = 128):
        super().__init__()
        chs = [base_channels * m for m in channel_mult]

        def _g(ch):
            for g in [8, 4, 2, 1]:
                if ch % g == 0:
                    return g
            return 1

        self.init_conv = nn.Conv3d(in_channels, chs[0], 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.enc_blocks.append(nn.Sequential(
                nn.GroupNorm(_g(chs[i]), chs[i]), nn.SiLU(),
                nn.Conv3d(chs[i], chs[i], 3, padding=1),
                nn.GroupNorm(_g(chs[i]), chs[i]), nn.SiLU(),
                nn.Conv3d(chs[i], chs[i + 1], 3, stride=2, padding=1),
            ))

        self.film_heads = nn.ModuleList()
        for ch in chs:
            head = nn.Sequential(
                nn.Linear(ch + time_emb_dim, ch * 2),
                nn.SiLU(),
                nn.Linear(ch * 2, ch * 2),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.film_heads.append(head)

        self.norms = nn.ModuleList([nn.GroupNorm(_g(ch), ch) for ch in chs])
        self.chs = chs

    def encode(self, image: torch.Tensor):
        image = image.to(dtype=self.init_conv.weight.dtype)
        feats = []
        h = self.init_conv(image)
        feats.append(h)
        for block in self.enc_blocks:
            h = block(h)
            feats.append(h)
        return feats

    def modulate(self, x, cond_feat, t_emb, scale_idx):
        cond_vec = cond_feat.mean(dim=[2, 3, 4])
        film_in  = torch.cat([cond_vec, t_emb], dim=1)
        gb       = self.film_heads[scale_idx](film_in)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma[:, :, None, None, None]
        beta  = beta [:, :, None, None, None]
        return (1 + gamma) * self.norms[scale_idx](x) + beta


# =====================================================================
# Model
# =====================================================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device   = time.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * (-np.log(10000) / (half_dim - 1)))
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, norm_groups=8, dropout=0.1):
        super().__init__()

        def _g(ch, pref=8):
            for g in [pref, 4, 2, 1]:
                if ch % g == 0:
                    return g
            return 1

        self.norm1    = nn.GroupNorm(_g(in_ch),  in_ch)
        self.act1     = nn.SiLU()
        self.conv1    = nn.Conv3d(in_ch,  out_ch, 3, padding=1)
        self.norm2    = nn.GroupNorm(_g(out_ch), out_ch)
        self.act2     = nn.SiLU()
        self.dropout  = nn.Dropout3d(dropout)
        self.conv2    = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch) if time_emb_dim else None
        self.skip     = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.conv1(self.act1(self.norm1(x)))
        if t_emb is not None and self.time_mlp is not None:
            h = h + self.time_mlp(self.act2(t_emb))[:, :, None, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock3D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm      = nn.GroupNorm(8 if channels % 8 == 0 else 4, channels)
        self.qkv       = nn.Conv3d(channels, channels * 3, 1)
        self.proj      = nn.Conv3d(channels, channels, 1)
        self.num_heads = num_heads
        self.scale     = (channels // num_heads) ** -0.5

    def forward(self, x):
        B, C, D, H, W = x.shape
        h   = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, D * H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out  = (attn @ v.transpose(-2, -1)).transpose(-2, -1).reshape(B, C, D, H, W)
        return x + self.proj(out)


class DownBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, attn=False, dropout=0.1):
        super().__init__()
        self.res1 = ResBlock3D(in_ch,  out_ch, time_emb_dim, dropout=dropout)
        self.res2 = ResBlock3D(out_ch, out_ch, time_emb_dim, dropout=dropout)
        self.attn = AttentionBlock3D(out_ch) if attn else nn.Identity()
        self.down = nn.Conv3d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attn(x)
        return self.down(x), x


class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, time_emb_dim, attn=False, dropout=0.1):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res1 = ResBlock3D(out_ch + skip_ch, out_ch, time_emb_dim, dropout=dropout)
        self.res2 = ResBlock3D(out_ch, out_ch, time_emb_dim, dropout=dropout)
        self.attn = AttentionBlock3D(out_ch) if attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return self.attn(x)


class SinUNet3D(nn.Module):

    def __init__(self, in_channels=2, out_channels=1, base_channels=24, channel_mult=(1, 2, 4, 8),
                 attn_levels=(False, False, False, True), time_emb_dim=128, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        chs = [base_channels * m for m in channel_mult]
        self.init_conv = nn.Conv3d(in_channels, chs[0], 3, padding=1)
        self.downs = nn.ModuleList([
            DownBlock3D(chs[i], chs[i + 1], time_emb_dim, attn_levels[i], dropout)
            for i in range(len(chs) - 1)
        ])
        self.mid_res1 = ResBlock3D(chs[-1], chs[-1], time_emb_dim, dropout=dropout)
        self.mid_attn = AttentionBlock3D(chs[-1])
        self.mid_res2 = ResBlock3D(chs[-1], chs[-1], time_emb_dim, dropout=dropout)
        self.ups = nn.ModuleList([
            UpBlock3D(chs[i + 1], chs[i], chs[i + 1], time_emb_dim, attn_levels[i], dropout)
            for i in reversed(range(len(chs) - 1))
        ])
        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv3d(chs[0], out_channels, 3, padding=1)

    def forward(self, x, timesteps, dce=None, cond_feats=None):
        t_emb = self.time_mlp(timesteps)
        h = self.init_conv(x)
        if dce is not None and cond_feats is not None:
            h = dce.modulate(h, cond_feats[0], t_emb, scale_idx=0)
        skips = []
        for i, down in enumerate(self.downs):
            h, skip = down(h, t_emb)
            if dce is not None and cond_feats is not None:
                skip = dce.modulate(skip, cond_feats[i + 1], t_emb, scale_idx=i + 1)
            skips.append(skip)
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)
        if dce is not None and cond_feats is not None:
            h = dce.modulate(h, cond_feats[3], t_emb, scale_idx=3)
        for up in self.ups:
            h = up(h, skips.pop(), t_emb)
        return self.out_conv(self.out_act(self.out_norm(h)))


class ConditionalDiffusionModel(nn.Module):
    def __init__(self, unet: SinUNet3D, dce: DynamicCondEncoder = None):
        super().__init__()
        self.unet = unet
        self.dce  = dce

    def forward(self, noisy_mask, timesteps, image_patch):
        x = torch.cat([noisy_mask, image_patch], dim=1)
        cond_feats = self.dce.encode(image_patch) if self.dce is not None else None
        return self.unet(x, timesteps, dce=self.dce, cond_feats=cond_feats)

    @torch.no_grad()
    def sample(self, image_patch, scheduler, num_inference_steps=50):
        scheduler.set_timesteps(num_inference_steps)
        x = torch.randn(
            (image_patch.shape[0], 1, *image_patch.shape[2:]),
            device=image_patch.device,
            dtype=image_patch.dtype,
        )
        cond_feats = self.dce.encode(image_patch) if self.dce is not None else None
        for t in scheduler.timesteps:
            t_batch = t.expand(x.shape[0]).to(x.device)
            inp     = torch.cat([x, image_patch], dim=1)
            pred    = self.unet(inp, t_batch, dce=self.dce, cond_feats=cond_feats)
            x = scheduler.step(pred, t, x).prev_sample
        return x.clamp(-1, 1)


def diffusion_output_to_prob(pred):
    """Map diffusion mask samples from [-1, 1] to foreground probability [0, 1]."""
    return ((pred + 1.0) * 0.5).clamp(0.0, 1.0)


def binary_entropy(prob, eps=UNCERTAINTY_EPS):
    prob = prob.clamp(eps, 1.0 - eps)
    return -(prob * torch.log(prob) + (1.0 - prob) * torch.log(1.0 - prob)) / np.log(2.0)


def _tensor_to_sitk_image(tensor, reference_img, dtype=np.float32):
    arr = tensor.detach().cpu().numpy().squeeze().transpose(2, 1, 0)
    img = sitk.GetImageFromArray(arr.astype(dtype))
    img.CopyInformation(reference_img)
    return img


def _safe_mask_mean(tensor, mask):
    if bool(mask.any().item()):
        return float(tensor[mask].mean().item())
    return 0.0


# =====================================================================
# Evaluation — one (scheduler, steps) config
# =====================================================================
@torch.no_grad()
def evaluate_one_config(
    data_loader, save_dir, model, scheduler, num_inference_steps,
    dice_metric, post_label, post_pred, n_samples=N_SAMPLES,
):
    """
    Run inference on all cases, record per-case timing and metrics.
    Returns aggregated results dict.
    """
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger()

    def _predict_patch(patch):
        return model.sample(patch, scheduler, num_inference_steps)

    model.eval()
    per_case = []

    for batch in data_loader:
        inputs   = batch["image"].to(device)
        labels   = batch["label_LA"].to(device)
        img_path = batch["image"].meta["filename_or_obj"][0]
        case_name = os.path.basename(os.path.dirname(img_path))

        # ---- timed inference ----
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        all_preds = []
        for _ in range(n_samples):
            pred = sliding_window_inference(
                inputs=inputs, roi_size=PATCH_SIZE, sw_batch_size=1,
                predictor=_predict_patch, overlap=0.5, mode="gaussian",
            )
            all_preds.append(pred)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        # -------------------------

        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_probs_tensor = diffusion_output_to_prob(all_preds_tensor)
        mean_pred        = all_preds_tensor.mean(dim=0, keepdim=True)
        mean_prob        = all_probs_tensor.mean(dim=0, keepdim=True)
        var_prob         = all_probs_tensor.var(dim=0, keepdim=True, unbiased=False)
        std_prob         = torch.sqrt(var_prob)
        entropy_map      = binary_entropy(mean_prob)

        preds_post  = [post_pred(p)  for p in decollate_batch(mean_pred)]
        labels_post = [post_label(l) for l in decollate_batch(labels)]

        dice_metric(y_pred=preds_post, y=labels_post)
        dice_score = dice_metric.aggregate().item()
        dice_metric.reset()

        original_img = sitk.ReadImage(img_path)
        spacing      = original_img.GetSpacing()
        pred_t  = torch.stack(preds_post,  dim=0)
        gt_t    = torch.stack(labels_post, dim=0)

        hd95 = compute_hausdorff_distance(pred_t, gt_t, spacing=spacing, percentile=95).mean().item()
        asd  = compute_average_surface_distance(pred_t, gt_t, spacing=spacing).mean().item()

        pred_mask = pred_t.bool()
        gt_mask   = gt_t.bool()
        high_entropy_mask = entropy_map >= 0.5
        uncertainty_scores = {
            "mean_std_all":        float(std_prob.mean().item()),
            "mean_var_all":        float(var_prob.mean().item()),
            "mean_entropy_all":    float(entropy_map.mean().item()),
            "max_entropy":         float(entropy_map.max().item()),
            "high_entropy_frac":   float(high_entropy_mask.float().mean().item()),
            "mean_std_pred":       _safe_mask_mean(std_prob, pred_mask),
            "mean_var_pred":       _safe_mask_mean(var_prob, pred_mask),
            "mean_entropy_pred":   _safe_mask_mean(entropy_map, pred_mask),
            "mean_std_gt":         _safe_mask_mean(std_prob, gt_mask),
            "mean_var_gt":         _safe_mask_mean(var_prob, gt_mask),
            "mean_entropy_gt":     _safe_mask_mean(entropy_map, gt_mask),
        }

        # save mean prediction and uncertainty maps
        sitk.WriteImage(
            _tensor_to_sitk_image(preds_post[0], original_img, dtype=np.uint8),
            os.path.join(save_dir, f"{case_name}_pred.nii.gz"),
        )
        sitk.WriteImage(
            _tensor_to_sitk_image(mean_prob, original_img, dtype=np.float32),
            os.path.join(save_dir, f"{case_name}_mean_prob.nii.gz"),
        )
        sitk.WriteImage(
            _tensor_to_sitk_image(std_prob, original_img, dtype=np.float32),
            os.path.join(save_dir, f"{case_name}_uncertainty_std.nii.gz"),
        )
        sitk.WriteImage(
            _tensor_to_sitk_image(var_prob, original_img, dtype=np.float32),
            os.path.join(save_dir, f"{case_name}_uncertainty_var.nii.gz"),
        )
        sitk.WriteImage(
            _tensor_to_sitk_image(entropy_map, original_img, dtype=np.float32),
            os.path.join(save_dir, f"{case_name}_uncertainty_entropy.nii.gz"),
        )

        if SAVE_INDIVIDUAL_SAMPLES:
            sample_dir = os.path.join(save_dir, f"{case_name}_samples")
            os.makedirs(sample_dir, exist_ok=True)
            for sample_idx, sample_pred in enumerate(decollate_batch(all_preds_tensor), start=1):
                sample_bin = post_pred(sample_pred)
                sitk.WriteImage(
                    _tensor_to_sitk_image(sample_bin, original_img, dtype=np.uint8),
                    os.path.join(sample_dir, f"sample_{sample_idx:03d}.nii.gz"),
                )

        logger.info(
            f"    [{case_name}]  Dice={dice_score:.4f}  HD95={hd95:.4f}  "
            f"ASD={asd:.4f}  Ent={uncertainty_scores['mean_entropy_pred']:.4f}  "
            f"Std={uncertainty_scores['mean_std_pred']:.4f}  Time={elapsed:.1f}s"
        )
        per_case.append({
            "case":          case_name,
            "dice":          float(dice_score),
            "hd95":          float(hd95),
            "asd":           float(asd),
            "inference_time_s": float(elapsed),
            **uncertainty_scores,
        })

    dice_vals = [c["dice"]             for c in per_case]
    hd95_vals = [c["hd95"]             for c in per_case]
    asd_vals  = [c["asd"]              for c in per_case]
    time_vals = [c["inference_time_s"] for c in per_case]
    ent_pred_vals = [c["mean_entropy_pred"] for c in per_case]
    std_pred_vals = [c["mean_std_pred"]     for c in per_case]
    var_pred_vals = [c["mean_var_pred"]     for c in per_case]
    ent_all_vals  = [c["mean_entropy_all"]  for c in per_case]
    std_all_vals  = [c["mean_std_all"]      for c in per_case]
    high_ent_vals = [c["high_entropy_frac"] for c in per_case]

    def _mean(vals):
        return float(np.mean(vals)) if vals else 0.0

    def _std(vals):
        return float(np.std(vals)) if vals else 0.0

    return {
        "per_case":        per_case,
        "mean_dice":       _mean(dice_vals),
        "std_dice":        _std(dice_vals),
        "mean_hd95":       _mean(hd95_vals),
        "std_hd95":        _std(hd95_vals),
        "mean_asd":        _mean(asd_vals),
        "std_asd":         _std(asd_vals),
        "mean_time_s":     _mean(time_vals),
        "std_time_s":      _std(time_vals),
        "total_time_s":    float(np.sum(time_vals)) if time_vals else 0.0,
        "mean_entropy_pred": _mean(ent_pred_vals),
        "std_entropy_pred":  _std(ent_pred_vals),
        "mean_std_pred":     _mean(std_pred_vals),
        "std_std_pred":      _std(std_pred_vals),
        "mean_var_pred":     _mean(var_pred_vals),
        "std_var_pred":      _std(var_pred_vals),
        "mean_entropy_all":  _mean(ent_all_vals),
        "std_entropy_all":   _std(ent_all_vals),
        "mean_std_all":      _mean(std_all_vals),
        "std_std_all":       _std(std_all_vals),
        "mean_high_entropy_frac": _mean(high_ent_vals),
        "std_high_entropy_frac":  _std(high_ent_vals),
    }


# =====================================================================
# Logger
# =====================================================================
def setup_logger(save_dir, log_filename="benchmark.log"):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger()


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":

    logger = setup_logger(results_dir)
    logger.info(f"Results will be saved to: {results_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Schedulers : {list(SCHEDULERS.keys())}")
    logger.info(f"Steps      : {INFERENCE_STEPS_LIST}")
    logger.info(f"N_SAMPLES  : {N_SAMPLES}")
    logger.info(f"Save individual samples: {SAVE_INDIVIDUAL_SAMPLES}")

    # ---- transforms ----
    val_transforms = Compose([
        LoadImaged(keys=["image", "label_LA"]),
        EnsureChannelFirstd(keys=["image", "label_LA"], channel_dim='no_channel'),
        Lambdad(keys=["label_LA"], func=lambda x: (x > 0).astype(np.float32)),
        Orientationd(keys=["image", "label_LA"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label_LA"], device=device),
    ])

    with open(json_path, "r", encoding="utf-8") as f:
        datalist = json.load(f)

    post_pred  = AsDiscrete(threshold=0.0)
    post_label = AsDiscrete(threshold=0.5)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    #num_folds = len([k for k in datalist if k.startswith("fold_")])
    num_folds = 1
    logger.info(f"Number of folds: {num_folds}")

    # ---- master results container ----
    benchmark_results = {
        "meta": {
            "model":       "CMDiff",
            "patch_size":  list(PATCH_SIZE),
            "n_samples":   N_SAMPLES,
            "save_individual_samples": SAVE_INDIVIDUAL_SAMPLES,
            "uncertainty_maps": [
                "mean_prob",
                "uncertainty_std",
                "uncertainty_var",
                "uncertainty_entropy",
            ],
            "schedulers":  list(SCHEDULERS.keys()),
            "steps_list":  INFERENCE_STEPS_LIST,
            "timestamp":   timestamp,
            "device":      str(device),
        },
        "folds": {}
    }

    for fold in range(num_folds):
        fold_key  = f"fold_{fold}"
        fold_name = f"fold_{fold + 1}"
        logger.info("\n" + "=" * 80)
        logger.info(f"FOLD {fold + 1}/{num_folds}")
        logger.info("=" * 80)

        fold_data  = datalist[fold_key]
        val_files  = fold_data.get("validation", [])
        logger.info(f"  Validation samples: {len(val_files)}")

        val_files_full = [
            {
                "image":      os.path.join(data_dir, item["image"]),
                "label_LA": os.path.join(data_dir, item["label_LA"]),
            }
            for item in val_files
        ]

        val_ds     = CacheDataset(data=val_files_full, transform=val_transforms, cache_num=4, cache_rate=1.0, num_workers=0)
        val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1, shuffle=False)

        # ---- build model once per fold ----
        #unet  = SinUNet3D(in_channels=2, base_channels=24).to(device)
        #model = ConditionalDiffusionModel(unet).to(device) # without dce 

        unet  = SinUNet3D(in_channels=2, base_channels=24)         
        dce   = DynamicCondEncoder(in_channels=1, base_channels=24) 
        model = ConditionalDiffusionModel(unet, dce).to(device) # with dce 
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  CMDiff trainable parameters: {total_params:,}")

        ckpt_path = os.path.join(model_dir, fold_name, f"best_diffmodel_fold{fold+1}_epoch3800.pth")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])  
        logger.info(f"  Loaded checkpoint: {ckpt_path}")

        fold_result = {"num_validation": len(val_files), "configs": {}}

        for sched_name, sched_factory in SCHEDULERS.items():
            fold_result["configs"][sched_name] = {}

            for n_steps in INFERENCE_STEPS_LIST:
                cfg_label = f"{sched_name}_steps{n_steps}"
                logger.info(f"\n  >> Scheduler={sched_name}  Steps={n_steps}")

                # fresh scheduler instance each time (avoids state leakage)
                scheduler = sched_factory()

                save_dir = os.path.join(results_dir, fold_name, cfg_label, "predictions")

                cfg_result = evaluate_one_config(val_loader, save_dir, model,scheduler,n_steps,dice_metric,post_label,post_pred,)
                cfg_result["scheduler"]         = sched_name
                cfg_result["num_inference_steps"] = n_steps

                fold_result["configs"][sched_name][n_steps] = cfg_result

                logger.info(
                    f"  Summary  Dice={cfg_result['mean_dice']:.4f}±{cfg_result['std_dice']:.4f}  "
                    f"HD95={cfg_result['mean_hd95']:.4f}±{cfg_result['std_hd95']:.4f}  "
                    f"ASD={cfg_result['mean_asd']:.4f}±{cfg_result['std_asd']:.4f}  "
                    f"Ent(pred)={cfg_result['mean_entropy_pred']:.4f}±{cfg_result['std_entropy_pred']:.4f}  "
                    f"Std(pred)={cfg_result['mean_std_pred']:.4f}±{cfg_result['std_std_pred']:.4f}  "
                    f"Time/case={cfg_result['mean_time_s']:.1f}±{cfg_result['std_time_s']:.1f}s"
                )

                # save per-config JSON immediately (fault-tolerant)
                per_cfg_json = os.path.join(results_dir, fold_name, cfg_label, "metrics.json")
                with open(per_cfg_json, "w") as f:
                    json.dump(cfg_result, f, indent=4)

        benchmark_results["folds"][fold_name] = fold_result

        # save fold-level JSON
        fold_json = os.path.join(results_dir, fold_name, "fold_benchmark.json")
        with open(fold_json, "w") as f:
            json.dump(fold_result, f, indent=4)

    # =====================================================================
    # Cross-fold summary table
    # =====================================================================
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK SUMMARY  (mean ± std across folds)")
    logger.info("=" * 80)

    summary_rows = []
    header = (
        f"{'Scheduler':<22} {'Steps':>6}  {'Dice':>14}  {'HD95':>14}  "
        f"{'ASD':>14}  {'Ent(pred)':>14}  {'Std(pred)':>14}  {'Time/case(s)':>14}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    cv_summary = {}
    for sched_name in SCHEDULERS:
        cv_summary[sched_name] = {}
        for n_steps in INFERENCE_STEPS_LIST:
            fold_metrics = []
            for fold_name, fold_data in benchmark_results["folds"].items():
                cfg = fold_data["configs"][sched_name].get(n_steps)
                if cfg:
                    fold_metrics.append(cfg)

            def _agg(key):
                vals = [m[key] for m in fold_metrics]
                return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)

            m_d, s_d = _agg("mean_dice")
            m_h, s_h = _agg("mean_hd95")
            m_a, s_a = _agg("mean_asd")
            m_e, s_e = _agg("mean_entropy_pred")
            m_u, s_u = _agg("mean_std_pred")
            m_t, s_t = _agg("mean_time_s")

            row = {
                "scheduler":          sched_name,
                "num_inference_steps": n_steps,
                "cv_mean_dice":       m_d, "cv_std_dice": s_d,
                "cv_mean_hd95":       m_h, "cv_std_hd95": s_h,
                "cv_mean_asd":        m_a, "cv_std_asd":  s_a,
                "cv_mean_entropy_pred": m_e, "cv_std_entropy_pred": s_e,
                "cv_mean_std_pred":     m_u, "cv_std_std_pred": s_u,
                "cv_mean_time_s":     m_t, "cv_std_time_s": s_t,
            }
            cv_summary[sched_name][n_steps] = row
            summary_rows.append(row)

            logger.info(
                f"{sched_name:<22} {n_steps:>6}  "
                f"{m_d:.4f}±{s_d:.4f}  "
                f"{m_h:.4f}±{s_h:.4f}  "
                f"{m_a:.4f}±{s_a:.4f}  "
                f"{m_e:.4f}±{s_e:.4f}  "
                f"{m_u:.4f}±{s_u:.4f}  "
                f"{m_t:.1f}±{s_t:.1f}"
            )

    benchmark_results["cv_summary"]  = cv_summary
    benchmark_results["summary_rows"] = summary_rows

    # final global JSON
    global_json = os.path.join(results_dir, "benchmark_summary.json")
    with open(global_json, "w") as f:
        json.dump(benchmark_results, f, indent=4)

    logger.info(f"\nAll results saved to: {results_dir}")
    logger.info(f"Global summary JSON : {global_json}")
