# Pure Multicoil SNRAware

This project keeps one production path: multicoil H5 preprocessing, ones-gmap
SNRAware input, learned physics correction, and LoRA/pre-post fine-tuning.

It intentionally does not include the previous single-coil retrofit, Monte
Carlo gmap estimation, ACS32/transfer/feasibility scripts, or baseline code.

## Data Flow

`multicoil h5 kspace -> image-domain crop/pad -> whitening -> coil compression
-> x8/cf0.04 mask and ACS -> GRAPPA -> SENSE combine -> scale -> [real, imag, ones-gmap]`

The clean target is the full-kspace SENSE-combined complex image from the same
whitened/compressed coil basis. The gmap channel is always ones in preprocessing;
`PhysicsCorrectionAdapter` learns bounded complex-scale and effective-gmap
corrections during fine-tuning.

## SOTA-Parity Invariants

The current fastMRI 5% random-volume run is intentionally aligned to the
working SOTA run in `/working2/arctic/snrawre/SNRAware`:

- SNRAware base model is built at `train.train_patch_size: [64, 64]`.
- Full validation images use sliding-window inference over the `384x384`
  preprocess crop with `train.overlap_for_inference: [16, 16, 0]`.
- Base checkpoint loading must report `matched_keys=1128`,
  `mismatched_keys=0`, and `total_model_keys=1128`.
- The public path keeps `preprocess.mc_gmap: 0` and
  `preprocess.gmap_mode: ones`; learned correction happens after the all-ones
  channel is assembled.

Do not change `train.train_patch_size` to the full crop size unless the base
checkpoint inheritance strategy is intentionally being changed.

## Main Commands

Dry run config resolution only:

```bash
cd /working2/arctic/project2/SNRAware
python train_multicoil.py \
  --config configs/multicoil/fastmri_x8_cf004_partial05_gmap_ones.yaml \
  --dry-run
```

Start training:

```bash
cd /working2/arctic/project2/SNRAware
python train_multicoil.py \
  --config configs/multicoil/fastmri_x8_cf004_partial05_gmap_ones.yaml
```

Custom H5 inference is isolated under `zero_shot/`:

```bash
cd /working2/arctic/project2/SNRAware
python zero_shot/run_zero_shot.py \
  --config configs/multicoil/template_generic_h5.yaml \
  --input-root /path/to/private_h5_or_dir \
  --output-dir /working2/arctic/project2/zero_shot_outputs
```

## Important Config Fields

- `preprocess.acc_factor`, `center_fraction`, `calib_center_fraction`: current
  fair default is x8 and 0.04/0.04.
- `preprocess.gmap_value`: always `1.0` for the training path.
- `preprocess.cache_dir`: transparent cache for preprocessed arrays and metadata.
- `correction`: bounds for learned complex scale and effective gmap correction.
- `lora`: regex-selected LoRA modules inside the frozen SNRAware base model.
- `subset`: optional training subset, with current public config using 5% random
  volume sampling.

## Cache Metadata

Each cached slice stores the arrays plus JSON metadata including source file,
source fingerprint, acceleration, center fraction, calibration fraction, mask
width, mask samples, ACS lines, whitening mode, coil counts, scale, and
`gmap_mode=ones_corrected`.

## Boundaries

Old repositories are read-only references:

- `/working2/arctic/snrawre/SNRAware`
- `/working2/arctic/unrolled_white`

This repo does not implement unrolled baselines or experiment comparison code.
