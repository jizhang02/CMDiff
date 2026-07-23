"""
MedSegDiff-style Training Script for MRI LA Segmentation
Diffusion-based segmentation under extreme class imbalance
"""

import os
import sys
import json
import logging
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import SimpleITK as sitk

# --- MONAI ---
from monai.data import CacheDataset, ThreadDataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.transforms import (
    AsDiscrete, Compose, EnsureChannelFirstd, EnsureTyped,
    LoadImaged, NormalizeIntensityd, Orientationd, Lambdad,
    RandAdjustContrastd, RandBiasFieldd, RandCropByPosNegLabeld,
    RandGaussianNoised, RandGaussianSmoothd, RandRotated,
    RandShiftIntensityd, RandZoomd, Rand3DElasticd,
)

# --- Diffusers ---
from diffusers import DDPMScheduler, DDIMScheduler

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (288, 192, 32)

root_dir = '/data/donnees/student1/jzhang/'
data_dir   = os.path.join(root_dir, "Lascarqs_stage1/train_data_cropped_130_384_256_40/") # LA path
json_path  = data_dir + "datalist_5fold.json"
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
results_dir = os.path.join(root_dir, f"LA_diff_results_{timestamp}")
model_dir   = os.path.join(root_dir, "LA_diff_results_2026_04_29_15_53/")

# =====================================================================
# Transforms
# =====================================================================
train_transforms = Compose([
    LoadImaged(keys=["image", "label_LA"]),
    EnsureChannelFirstd(keys=["image", "label_LA"], channel_dim='no_channel'),
    Lambdad(keys=["label_LA"], func=lambda x: (x > 0).astype(np.float32)),
    Orientationd(keys=["image", "label_LA"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandBiasFieldd(keys="image", prob=0.15, coeff_range=(0.02, 0.06), degree=2),
    Rand3DElasticd(keys=["image", "label_LA"], prob=0.1, sigma_range=(3, 5), magnitude_range=(2, 6)),
    RandRotated(keys=["image", "label_LA"], prob=0.5, range_x=0.26, range_y=0.26, range_z=0.26,
                mode=["bilinear", "nearest"]),
    RandZoomd(keys=["image", "label_LA"], min_zoom=0.9, max_zoom=1.1, prob=0.3),
    RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 2.0)),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.3)),
    RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.02),
    RandCropByPosNegLabeld(keys=["image", "label_LA"], label_key="label_LA", spatial_size=PATCH_SIZE, pos=2, neg=2,
        num_samples=4, image_key="image", image_threshold=0, allow_smaller=True,),
    EnsureTyped(keys=["image", "label_LA"], dtype=torch.float32, device="cpu", track_meta=False),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label_LA"]),
    EnsureChannelFirstd(keys=["image", "label_LA"], channel_dim='no_channel'),
    Lambdad(keys=["label_LA"], func=lambda x: (x > 0).astype(np.float32)),
    Orientationd(keys=["image", "label_LA"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label_LA"], dtype=torch.float32, device="cpu", track_meta=True),
])

# =====================================================================
# Dynamic Conditional Encoder (DCE)
# use_checkpoint=True: gradient checkpointing on enc_blocks (like MONAI)
# =====================================================================
class DynamicCondEncoder(nn.Module):

    def __init__(self, in_channels: int = 1, base_channels: int = 24,
                 channel_mult: tuple = (1, 2, 4, 8), time_emb_dim: int = 128,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        chs = [base_channels * m for m in channel_mult]  # [24, 48, 96, 192]

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
        feats.append(h)                         # scale 0
        for block in self.enc_blocks:
            if self.use_checkpoint:
                h = gradient_checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
            feats.append(h)                     # scale 1, 2, 3
        return feats                            # len = 4

    def modulate(self, x: torch.Tensor, cond_feat: torch.Tensor, t_emb: torch.Tensor, scale_idx: int) -> torch.Tensor:
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
    """
    use_checkpoint: if True, res1 and res2 use gradient checkpointing.
    AttentionBlock is NOT checkpointed (low memory footprint, no t_emb dependency).
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, attn=False, dropout=0.1,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.res1 = ResBlock3D(in_ch,  out_ch, time_emb_dim, dropout=dropout)
        self.res2 = ResBlock3D(out_ch, out_ch, time_emb_dim, dropout=dropout)
        self.attn = AttentionBlock3D(out_ch) if attn else nn.Identity()
        self.down = nn.Conv3d(out_ch, out_ch, 3, stride=2, padding=1)

    def _ckpt(self, module, x, t_emb):
        """Wrapper so checkpoint receives only Tensors as inputs."""
        return gradient_checkpoint(module, x, t_emb, use_reentrant=False)

    def forward(self, x, t_emb):
        if self.use_checkpoint:
            x = self._ckpt(self.res1, x, t_emb)
            x = self._ckpt(self.res2, x, t_emb)
        else:
            x = self.res1(x, t_emb)
            x = self.res2(x, t_emb)
        x = self.attn(x)
        return self.down(x), x          # (downsampled, skip)


class UpBlock3D(nn.Module):
    """
    use_checkpoint: if True, res1 and res2 use gradient checkpointing.
    """
    def __init__(self, in_ch, out_ch, skip_ch, time_emb_dim, attn=False, dropout=0.1,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res1 = ResBlock3D(out_ch + skip_ch, out_ch, time_emb_dim, dropout=dropout)
        self.res2 = ResBlock3D(out_ch, out_ch, time_emb_dim, dropout=dropout)
        self.attn = AttentionBlock3D(out_ch) if attn else nn.Identity()

    def _ckpt(self, module, x, t_emb):
        return gradient_checkpoint(module, x, t_emb, use_reentrant=False)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        if self.use_checkpoint:
            x = self._ckpt(self.res1, x, t_emb)
            x = self._ckpt(self.res2, x, t_emb)
        else:
            x = self.res1(x, t_emb)
            x = self.res2(x, t_emb)
        return self.attn(x)


class SinUNet3D(nn.Module):
    """
    use_checkpoint: propagated to all DownBlock3D, UpBlock3D, and mid ResBlocks.
    Mirrors MONAI's pattern: pass use_checkpoint once at the top-level model.
    """
    def __init__(self, in_channels=2, out_channels=1, base_channels=24,
                 channel_mult=(1, 2, 4, 8),
                 attn_levels=(False, False, False, True),
                 time_emb_dim=128, dropout=0.1,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        chs = [base_channels * m for m in channel_mult]   # [24,48,96,192]
        self.init_conv = nn.Conv3d(in_channels, chs[0], 3, padding=1)

        self.downs = nn.ModuleList([
            DownBlock3D(chs[i], chs[i + 1], time_emb_dim,
                        attn_levels[i], dropout,
                        use_checkpoint=use_checkpoint)
            for i in range(len(chs) - 1)
        ])

        self.mid_res1 = ResBlock3D(chs[-1], chs[-1], time_emb_dim, dropout=dropout)
        self.mid_attn = AttentionBlock3D(chs[-1])
        self.mid_res2 = ResBlock3D(chs[-1], chs[-1], time_emb_dim, dropout=dropout)

        self.ups = nn.ModuleList([
            UpBlock3D(chs[i + 1], chs[i], chs[i + 1], time_emb_dim,
                      attn_levels[i], dropout,
                      use_checkpoint=use_checkpoint)
            for i in reversed(range(len(chs) - 1))
        ])

        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv3d(chs[0], out_channels, 3, padding=1)

    def _ckpt_mid(self, module, x, t_emb):
        return gradient_checkpoint(module, x, t_emb, use_reentrant=False)

    def forward(self, x, timesteps, dce: DynamicCondEncoder = None, cond_feats: list = None):
        t_emb = self.time_mlp(timesteps)
        h = self.init_conv(x)

        if dce is not None and cond_feats is not None:
            h = dce.modulate(h, cond_feats[0], t_emb, scale_idx=0)

        skips = []
        for i, down in enumerate(self.downs):
            h, skip = down(h, t_emb)           # checkpointing handled inside DownBlock3D
            if dce is not None and cond_feats is not None:
                skip = dce.modulate(skip, cond_feats[i + 1], t_emb, scale_idx=i + 1)
            skips.append(skip)

        # Mid blocks — checkpoint if enabled
        if self.use_checkpoint:
            h = self._ckpt_mid(self.mid_res1, h, t_emb)
        else:
            h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)                   # AttentionBlock: not checkpointed
        if self.use_checkpoint:
            h = self._ckpt_mid(self.mid_res2, h, t_emb)
        else:
            h = self.mid_res2(h, t_emb)

        if dce is not None and cond_feats is not None:
            h = dce.modulate(h, cond_feats[3], t_emb, scale_idx=3)

        for up in self.ups:
            h = up(h, skips.pop(), t_emb)      # checkpointing handled inside UpBlock3D

        return self.out_conv(self.out_act(self.out_norm(h)))


# =====================================================================
# ConditionalDiffusionModel
# =====================================================================
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
        x = torch.randn((image_patch.shape[0], 1, *image_patch.shape[2:]), device=image_patch.device, dtype=image_patch.dtype,)
        cond_feats = self.dce.encode(image_patch) if self.dce is not None else None
        for t in scheduler.timesteps:
            t_batch = t.expand(x.shape[0]).to(x.device)
            inp     = torch.cat([x, image_patch], dim=1)
            x = scheduler.step(self.unet(inp, t_batch, dce=self.dce, cond_feats=cond_feats), t, x).prev_sample
        return x.clamp(-1, 1)
        

# =====================================================================
# Training
# =====================================================================
def train_one_epoch(loader, model, ema_model, ddpm_scheduler, optimizer, lr_sched, scaler):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    alphas_cumprod = ddpm_scheduler.alphas_cumprod.to(device)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True).float()
        labels = batch["label_LA"].to(device, non_blocking=True).float()

        B = images.shape[0]
        timesteps    = torch.randint(0, ddpm_scheduler.config.num_train_timesteps, (B,), device=device).long()
        noise        = torch.randn_like(labels)
        labels_m11   = labels * 2 - 1        # [0,1] → [-1,1]
        noisy_labels = ddpm_scheduler.add_noise(labels_m11, noise, timesteps)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred_x0 = model(noisy_labels, timesteps, images)

            # L_diff：SNR MSE
            snr       = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps]).clamp(min=1e-8)
            snr_weight = (snr / (snr + 1)).clamp(0, 1)[:, None, None, None, None]
            mse       = F.mse_loss(pred_x0, labels_m11, reduction='none')
            L_diff    = (mse * snr_weight).mean()

            # L_seg：Dice + BCE
            pred_x0    = torch.clamp(pred_x0, -1, 1)
            pred_logits = pred_x0
            pred_prob   = torch.sigmoid(pred_logits)

            pred_flat  = pred_prob.view(B, -1)
            label_flat = labels.view(B, -1)

            intersection = (pred_flat * label_flat).sum(dim=1)
            denom        = pred_flat.sum(dim=1) + label_flat.sum(dim=1)
            dice_loss    = 1 - (2.0 * intersection + 1e-5) / (denom + 1e-5)
            dice_loss    = torch.nan_to_num(dice_loss, nan=1.0)

            bce     = F.binary_cross_entropy_with_logits(pred_logits, labels, reduction='none')
            weights = torch.where(labels > 0.5, 100.0, 1.0)
            bce     = (bce * weights).view(B, -1).mean(dim=1)

            loss_per_sample = 0.8 * dice_loss + 0.2 * bce
            t_weight = 1.0 - timesteps.float() / ddpm_scheduler.config.num_train_timesteps
            L_seg    = (t_weight * loss_per_sample).mean()
            L_seg    = torch.nan_to_num(L_seg, nan=0.0)

            loss = L_diff + 0.5 * L_seg

        with torch.no_grad():
            pred_binary  = (pred_x0 > 0).float()
            intersection = (pred_binary * labels).sum()
            denom        = pred_binary.sum() + labels.sum()
            train_dice   = (2.0 * intersection / (denom + 1e-5)).item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        ema_model.update_parameters(model)
        lr_sched.step()

        total_loss += loss.item()
        total_dice += train_dice

    return total_loss / max(len(loader), 1), total_dice / max(len(loader), 1)


# =====================================================================
# Validation
# =====================================================================
@torch.no_grad()
def validation(loader, model, ddim_scheduler, dice_metric, post_label, post_pred, num_infer_step=20):
    model.eval()

    def _pred(patch):
        return model.sample(patch, ddim_scheduler, num_inference_steps=num_infer_step)

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label_LA"].to(device)
        preds  = sliding_window_inference(images, PATCH_SIZE, 1, _pred, overlap=0.5, mode="gaussian")
        dice_metric(y_pred=[post_pred(p) for p in decollate_batch(preds)], y =[post_label(l) for l in decollate_batch(labels)],)

    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    return metric


# =====================================================================
# Logger
# =====================================================================
def setup_logger(save_dir, log_filename="train.log"):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger()


# =====================================================================
# Main
# =====================================================================
def main(args):
    logger = setup_logger(args.save_dir)

    post_pred  = AsDiscrete(threshold=0.0)
    post_label = AsDiscrete(threshold=0.5)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    num_epochs   = 4000
    initial_lr   = 1e-4
    dce_lr       = 3e-4   
    resume_epoch = 2000   
    resume       = True   

    with open(json_path, "r", encoding="utf-8") as f:
        datalist = json.load(f)

    num_folds = 1
    logger.info(f"Number of folds: {num_folds}")

    for fold in range(num_folds):
        fold_key = f"fold_{fold}"
        logger.info("\n" + "=" * 70)
        logger.info(f"FOLD {fold + 1}/{num_folds}")
        logger.info("=" * 70)

        fold_dir             = os.path.join(results_dir, f"fold_{fold + 1}")
        fold_predictions_dir = os.path.join(fold_dir, "predictions")
        os.makedirs(fold_predictions_dir, exist_ok=True)

        fold_data   = datalist[fold_key]
        train_files = fold_data.get("training",   [])
        val_files   = fold_data.get("validation", [])

        logger.info(f"  Training samples:   {len(train_files)}")
        logger.info(f"  Validation samples: {len(val_files)}")

        train_files_full = [
            {"image": os.path.join(data_dir, item["image"]),
             "label_LA": os.path.join(data_dir, item["label_LA"])}
            for item in train_files
        ]
        val_files_full = [
            {"image": os.path.join(data_dir, item["image"]),
             "label_LA": os.path.join(data_dir, item["label_LA"])}
            for item in val_files
        ]

        train_ds     = CacheDataset(data=train_files_full, transform=train_transforms, cache_num=8, cache_rate=0.0, num_workers=0)
        train_loader = ThreadDataLoader(train_ds, num_workers=4, batch_size=1, shuffle=True)

        val_ds     = CacheDataset(data=val_files_full, transform=val_transforms, cache_num=4, cache_rate=1.0, num_workers=0)
        val_loader = ThreadDataLoader(val_ds, num_workers=4, batch_size=1, shuffle=False)

        # Build model — pass use_checkpoint to both UNet and DCE
        unet  = SinUNet3D(in_channels=2, base_channels=24, use_checkpoint=True) 
        dce   = DynamicCondEncoder(in_channels=1, base_channels=24, use_checkpoint=True) # with dce
        model = ConditionalDiffusionModel(unet, dce).to(device)
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9999))

        optimizer = torch.optim.AdamW([
            {"params": model.unet.parameters(), "lr": initial_lr},
            {"params": model.dce.parameters(),  "lr": dce_lr},
        ], weight_decay=1e-5)

        scaler = torch.GradScaler("cuda")
        warmup_iters  = 500
        total_iters   = num_epochs * len(train_loader)
        warmup_sched  = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters)
        cosine_sched  = CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters, eta_min=1e-6)
        lr_sched      = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_iters])

        dice_val_best = 0.0
        best_epoch    = 0
        start_epoch   = resume_epoch + 1 if resume else 1

        if resume: # load model
            ckpt_path = os.path.join(model_dir, f'fold_{fold+1}', f"best_diffmodel_fold{fold+1}.pth")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model"])            
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            dice_val_best = checkpoint["dice_val_best"]            
            ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9999))
            already_done_steps = resume_epoch * len(train_loader)
            for _ in range(already_done_steps):
                lr_sched.step()
            logger.info(f"✓ Continue training, loaded checkpoint from {ckpt_path}")
            logger.info(f"✓ LR scheduler fast-forwarded {already_done_steps} steps")     

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        unet_params  = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
        dce_params   = sum(p.numel() for p in model.dce.parameters()  if p.requires_grad)
        logger.info(f"Total trainable parameters: {total_params:,}")
        logger.info(f"  UNet : {unet_params:,}")
        logger.info(f"  DCE  : {dce_params:,}")

        # --- Diffusion Schedulers ---
        ddpm_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0005, beta_end=0.0195, beta_schedule="linear", prediction_type="sample",)
        ddim_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.0005, beta_end=0.0195, beta_schedule="linear", clip_sample=True, prediction_type="sample",)

        logger.info(f"\nTraining Fold {fold + 1}...")

        for epoch in range(start_epoch, num_epochs + 1):
            train_loss, train_dice = train_one_epoch(train_loader, model, ema_model, ddpm_scheduler, optimizer, lr_sched, scaler)

            if epoch % 10 == 0:
                logger.info("Validating...")
                valid_dice = validation(val_loader, ema_model.module, ddim_scheduler, dice_metric, post_label, post_pred, num_infer_step=20,)
                logger.info(f"Epoch {epoch}/{num_epochs}  Train Loss: {train_loss:.5f}  Train Dice: {train_dice:.5f} Val Dice: {valid_dice:.5f}")
                if valid_dice > dice_val_best:
                    dice_val_best = valid_dice
                    best_epoch    = epoch
                    torch.save({
                        "model":         ema_model.module.state_dict(),
                        "optimizer":     optimizer.state_dict(),
                        "scaler":        scaler.state_dict(),
                        "epoch":         epoch,
                        "dice_val_best": dice_val_best,
                    }, os.path.join(fold_dir, f"best_diffmodel_fold{fold+1}.pth"))
                    logger.info(f"  ✓ Model saved (Val Dice={dice_val_best:.5f})")
            else:
                logger.info(f"Epoch {epoch}/{num_epochs} Train Loss: {train_loss:.5f}  Train Dice: {train_dice:.5f}")

        logger.info(f"\nBest Val Dice: {dice_val_best:.4f} @ epoch {best_epoch}")

    print("=" * 70)
    print("\nAll cross-validation tasks completed successfully!")
    print(f"Results directory: {results_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=results_dir)
    main(parser.parse_args())