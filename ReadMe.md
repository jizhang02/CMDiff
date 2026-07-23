# CMDiff

CMDiff is the research code accompanying our feasibility study of conditional
diffusion models for 3D left-atrial (LA) and LA-scar segmentation in
late-gadolinium-enhanced magnetic resonance imaging (LGE-MRI).

The repository provides:

- training scripts for LA and scar segmentation;
- evaluation with four diffusion samplers and several sampling budgets;
- repeated-sampling analysis of prediction stability;
- NIfTI predictions and case-level Dice, HD95, ASD, and inference-time results.

> **Research-code status.** These files preserve the configurations used for
> the experiments, including author-specific paths, GPU indices, fold
> selections, and checkpoint names. They are not yet a fully parameterised
> command-line package. Read **Configuration required before running** before
> starting an experiment.

## Repository contents

| File | Function | Typical use |
|---|---|---|
| `LA_diffusion.py` | Trains the 3D LGE-conditioned diffusion model to predict the LA mask. | Train or resume an LA model and save the checkpoint with the best validation Dice. |
| `LA_diffusion_evaluate.py` | Benchmarks a trained LA model using DDIM, DPM-Solver Multistep, Euler, and UniPC samplers. | Compare sampling methods and 20/50/100/200 inference steps; export masks and case-level metrics. |
| `LA_diffusion_uncertainty_evalulate.py` | Repeats LA inference and calculates voxelwise prediction-dispersion maps. | Generate mean probability, standard deviation, variance, and entropy maps from repeated samples. The filename retains the original `evalulate` spelling. |
| `scar_diffusion.py` | Trains the same conditional-diffusion framework with scar masks as the target. | Train or resume a scar model and select the checkpoint with the best validation Dice. |
| `scar_diffusion_evaluate.py` | Benchmarks a trained scar model across samplers and sampling budgets. | Export scar masks and calculate case-level Dice, HD95, ASD, and inference time. |
| `scar_diffusion_uncertainty_evaluate.py` | Repeats scar inference and measures prediction dispersion. | Generate scar probability and stability maps and their summary statistics. |
| `ReadMe.md` | Documents the code, data format, configuration, commands, and outputs. | Start here before running any script. |

The model classes are defined inside each script so that every experiment file
is self-contained. Consequently, some architecture, transform, inference, and
metric code is duplicated.

## Method summary

All training and evaluation scripts use a 3D U-Net-like denoiser conditioned on
multiscale LGE-MRI features from a DCE encoder. The released configuration:

- predicts the clean segmentation mask directly
  (`prediction_type="sample"`), rather than predicting diffusion noise;
- maps a binary target from `[0, 1]` to `[-1, 1]` during diffusion training;
- uses a DDPM scheduler for training and iterative reverse diffusion at
  inference;
- uses a patch size of `(288, 192, 32)`;
- converts the image and label orientation to RAS;
- normalises LGE intensities over nonzero voxels;
- applies spatial and intensity augmentation during training;
- uses AdamW, mixed precision, exponential moving average (EMA), gradient
  clipping, and gradient checkpointing.

The evaluation scripts convert the final diffusion output into a binary mask
with a threshold of `0`, which corresponds to `0.5` after mapping the output
from `[-1, 1]` to `[0, 1]`.

## Requirements

The code was designed for Linux with an NVIDIA GPU. An A6000-class GPU is
recommended for the released patch size. CPU fallback is present but is not
practical for training or repeated diffusion inference.

Main Python dependencies:

- Python 3.10 or a compatible version;
- PyTorch with CUDA support;
- MONAI;
- Hugging Face `diffusers`;
- SimpleITK;
- NumPy.

A minimal environment can be created as follows:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the PyTorch build appropriate for the CUDA version on your system,
# then install the remaining packages:
python -m pip install monai diffusers SimpleITK numpy
```

For exact reproduction, record the Python, PyTorch, CUDA, MONAI, and
`diffusers` versions used in each run. The scripts set random seeds to `42`,
but exact bitwise reproducibility across GPUs and library versions is not
guaranteed.

## Data preparation

The dataset is not distributed with this repository. Obtain the LGE-MRI data
and annotations from their original source and comply with the corresponding
data-use terms.

### LA data

`LA_diffusion.py` and the two LA evaluation scripts expect a five-fold JSON
manifest named `datalist_5fold.json`. Paths in the manifest are relative to
`data_dir`.

Example directory:

```text
train_data_cropped_130_384_256_40/
├── datalist_5fold.json
├── train_01/
│   ├── enhanced.nii.gz
│   └── atriumSegImgMO.nii.gz
└── ...
```

Example manifest structure:

```json
{
  "fold_0": {
    "training": [
      {
        "image": "train_01/enhanced.nii.gz",
        "label_LA": "train_01/atriumSegImgMO.nii.gz"
      }
    ],
    "validation": []
  }
}
```

The original five-fold split contains 104 training cases and 26 validation
cases per fold.

### Scar data

`scar_diffusion.py` and the two scar evaluation scripts expect
`datalist_5fold_scarfixed.json`.

Example directory:

```text
train_data_cropped_60_384_256_40/
├── datalist_5fold_scarfixed.json
├── train_02/
│   ├── enhanced.nii.gz
│   ├── atriumSegImgMO_wall.nii.gz
│   └── scarSegImgM_fixed.nii.gz
└── ...
```

Example manifest structure:

```json
{
  "fold_0": {
    "training": [
      {
        "image": "train_02/enhanced.nii.gz",
        "label_LA": "train_02/atriumSegImgMO_wall.nii.gz",
        "label_scar": "train_02/scarSegImgM_fixed.nii.gz"
      }
    ],
    "validation": []
  }
}
```

The original five-fold split contains 48 training cases and 12 validation
cases per fold. The current scar scripts read `image` and `label_scar`;
`label_LA` may be retained in the manifest for provenance, but it is not used
as an input by these scripts.

Labels are binarised internally with `label > 0`. Images should already be
cropped and paired correctly with their labels. The scripts handle RAS
orientation and nonzero-voxel intensity normalisation, but they do not perform
case matching, registration, or dataset-integrity checks.

## Configuration required before running

Edit the configuration block near the beginning of each selected script.

1. Set `root_dir`, `data_dir`, and `json_path` to the local dataset.
2. For training, set `results_dir` to a new output directory.
3. For evaluation, set `model_dir` to the trained run and `results_dir` to a
   new evaluation directory.
4. Change `cuda:0` or `cuda:1` to the available GPU, for example:

   ```python
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   ```

5. Check the fold loop. The released files retain the subsets used in
   individual experiments:

   - `LA_diffusion.py` trains only `fold_0`;
   - `LA_diffusion_evaluate.py` evaluates only `fold_0`;
   - `LA_diffusion_uncertainty_evalulate.py` evaluates only `fold_0`;
   - `scar_diffusion.py` trains only `fold_2`;
   - `scar_diffusion_evaluate.py` evaluates only `fold_2`;
   - `scar_diffusion_uncertainty_evaluate.py` evaluates only `fold_0`.

   To run every fold in a five-fold manifest, use:

   ```python
   num_folds = len([key for key in datalist if key.startswith("fold_")])
   for fold in range(num_folds):
       ...
   ```

6. For a fresh LA run, change:

   ```python
   resume = False
   resume_epoch = 0
   ```

   The released `LA_diffusion.py` otherwise attempts to resume from epoch
   `2000`. Scar training already defaults to a fresh run.

7. When resuming or evaluating, make the checkpoint name in the script match
   the file on disk. In particular, the LA evaluation scripts currently expect:

   ```text
   fold_1/best_diffmodel_fold1_epoch3800.pth
   ```

   whereas `LA_diffusion.py` normally saves:

   ```text
   fold_1/best_diffmodel_fold1.pth
   ```

> **Important:** the training option `--save_dir` currently changes the log
> destination only. Checkpoints are still written below the global
> `results_dir` defined inside the script. Therefore, set `results_dir` in the
> file before training; do not rely only on `--save_dir`.

## Training

Run the commands from this directory:

```bash
cd deepscar/code_archive_diffusion/CMDiff
```

### Train the LA model

After setting the paths, fold loop, and `resume` options:

```bash
python LA_diffusion.py
```

The default configuration trains for 4000 epochs and validates every 10
epochs. The checkpoint with the highest validation Dice is saved as:

```text
<results_dir>/fold_1/best_diffmodel_fold1.pth
```

Each additional fold is saved under its corresponding `fold_N` directory.

### Train the scar model

After setting the paths and fold loop:

```bash
python scar_diffusion.py
```

The default configuration trains for 3000 epochs and validates every 10
epochs. Its best checkpoint is saved as:

```text
<results_dir>/fold_N/best_diffmodel_foldN.pth
```

Training validation uses DDIM with 20 reverse-diffusion steps. Checkpoints
contain the model, optimiser, mixed-precision scaler, epoch, and best validation
Dice. Keep each checkpoint paired with the exact fold manifest and source-code
configuration that produced it.

## Standard evaluation

### Evaluate LA segmentation

Set `data_dir`, `json_path`, `model_dir`, `results_dir`, the fold loop, and the
checkpoint filename in `LA_diffusion_evaluate.py`, then run:

```bash
python LA_diffusion_evaluate.py
```

### Evaluate scar segmentation

Configure `scar_diffusion_evaluate.py`, then run:

```bash
python scar_diffusion_evaluate.py
```

Both scripts compare:

- `DDIMScheduler`;
- `DPMSolverMultistepScheduler`;
- `EulerDiscreteScheduler`;
- `UniPCMultistepScheduler`;
- 20, 50, 100, and 200 reverse-diffusion steps;
- three stochastic predictions per case (`N_SAMPLES = 3`).

The three continuous predictions are averaged before thresholding. For every
case and configuration, the scripts report:

- Dice coefficient;
- 95th-percentile Hausdorff distance (HD95);
- average surface distance (ASD);
- inference time.

HD95 and ASD use the original NIfTI spacing. The saved JSON standard deviations
are NumPy population standard deviations (`ddof=0`).

## Repeated-sampling stability analysis

### Analyse LA predictions

Configure `LA_diffusion_uncertainty_evalulate.py`, then run:

```bash
python LA_diffusion_uncertainty_evalulate.py
```

The current LA script uses 20 repeated samples and evaluates 20 and 50
inference steps.

### Analyse scar predictions

Configure `scar_diffusion_uncertainty_evaluate.py`, then run:

```bash
python scar_diffusion_uncertainty_evaluate.py
```

The current scar script uses 20 repeated samples and evaluates 20, 50, 100,
and 200 inference steps.

For each case, the uncertainty scripts map every diffusion output to `[0, 1]`
and save:

- `*_pred.nii.gz`: thresholded mean prediction;
- `*_mean_prob.nii.gz`: mean probability across samples;
- `*_uncertainty_std.nii.gz`: voxelwise standard deviation;
- `*_uncertainty_var.nii.gz`: voxelwise variance;
- `*_uncertainty_entropy.nii.gz`: binary entropy in bits, normalised to
  `[0, 1]`.

They also record uncertainty summaries over the complete image, predicted
foreground, and reference foreground. Set
`SAVE_INDIVIDUAL_SAMPLES = True` only when the individual stochastic masks are
needed, because this substantially increases storage use.

Entropy, variance, and standard deviation in this repository measure
disagreement among repeated predictions from the same model. They describe
prediction stability; they are not, by themselves, validated measures of
annotation disagreement or clinical uncertainty. Interpret them together with
case-level segmentation accuracy.

## Output structure

A typical evaluation run produces:

```text
<results_dir>/
├── benchmark.log
├── benchmark_summary.json
└── fold_1/
    ├── fold_benchmark.json
    └── DDIMScheduler_steps20/
        ├── metrics.json
        └── predictions/
            ├── <case>_pred.nii.gz
            └── ...
```

Uncertainty runs add the probability, standard-deviation, variance, and
entropy NIfTI files listed above. Predictions retain the geometry of the input
image.

## Computational cost

Diffusion evaluation can be substantially slower than a deterministic
segmentation network. Its approximate workload scales as:

```text
number of cases × number of samplers × sampling-step settings × N_SAMPLES
```

Before launching the complete benchmark, test one fold, one sampler, 20 steps,
and one case. After confirming paths, checkpoint compatibility, GPU memory, and
saved geometry, expand to the intended experiment.

## Recommended reproducible workflow

1. Freeze and archive the five-fold manifest.
2. Train one model independently for every fold.
3. Use only the corresponding held-out validation cases for that fold.
4. Compare samplers and step counts on the predefined development data.
5. Freeze the selected inference settings before evaluating any untouched test
   data.
6. Run repeated-sampling analysis using the same checkpoint and held-out cases.
7. Report case-level results in addition to aggregate means and standard
   deviations.

Do not combine a prediction with a checkpoint trained on that case when
claiming out-of-fold performance.

## Citation and code availability

The manuscript citation and public repository URL will be added after the
arXiv record and repository are available:

```text
Paper: [arXiv link to be added]
Code:  [repository link to be added]
```

Before public release, also add a software licence and, if checkpoints are
shared, document their training folds and permitted terms of use.

## Disclaimer

This software is released for research and reproducibility purposes. It is not
a medical device and must not be used for clinical decision-making without
appropriate independent validation.
