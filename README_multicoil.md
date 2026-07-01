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

## Depth Defaults

The multicoil path is 2D-first while keeping the unified tensor contract
`[B,C,D,H,W]`:

- Default training and validation patch depth is `train.patch.depth: 1`.
- Explicit 3D is supported only with `train.patch.depth: 16`.
- Other patch depths are rejected at config load time.
- Full validation images still use sliding-window inference over the `384x384`
  preprocess crop and aggregate full-volume PSNR/SSIM/NMSE.

The current fastMRI 5% random-volume config keeps these invariants:

- SNRAware base model is built at `train.patch: {depth: 1, height: 64, width: 64}`
  by default.
- Full validation images use sliding-window inference over the `384x384`
  preprocess crop with `train.inference_overlap: {depth: 0, height: 16, width: 16}`.
- Base checkpoint loading must report `matched_keys=1128`,
  `mismatched_keys=0`, and `total_model_keys=1128`.
- The public path keeps `preprocess.mc_gmap: 0` and
  `preprocess.gmap_mode: ones`; learned correction happens after the all-ones
  channel is assembled.

Do not change `train.patch.height` or `train.patch.width` to the full crop size
unless the base checkpoint inheritance strategy is intentionally being changed.

## Base Model Checkpoints

Pretrained SNRAware weights are not committed to this repository. Download the
public checkpoints from Hugging Face and place them under `checkpoints/`:

```bash
cd /working2/arctic/project2/SNRAware
mkdir -p checkpoints/large checkpoints/small

wget -P checkpoints/large \
  https://huggingface.co/microsoft/SNRAware/resolve/main/large/snraware_large_model.pts
wget -P checkpoints/large \
  https://huggingface.co/microsoft/SNRAware/resolve/main/large/snraware_large_model.yaml

wget -P checkpoints/small \
  https://huggingface.co/microsoft/SNRAware/resolve/main/small/snraware_small_model.pts
wget -P checkpoints/small \
  https://huggingface.co/microsoft/SNRAware/resolve/main/small/snraware_small_model.yaml
```

The active config only stores `base_model.variant`. At runtime, `large` resolves
to `checkpoints/large/snraware_large_model.{yaml,pts}`, and `small` resolves to
`checkpoints/small/snraware_small_model.{yaml,pts}`.

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

Start the explicit 3D-16 path:

```bash
cd /working2/arctic/project2/SNRAware
python train_multicoil.py \
  --config configs/multicoil/fastmri_x8_cf004_partial05_gmap_ones.yaml \
  --set train.patch.depth=16 \
  --set train.inference_overlap.depth=8 \
  --set runtime.run_name=fastmri_x8_cf004_partial05_3d_d16_gmap_ones
```

Switch the inherited SNRAware backbone to the local small checkpoint:

```bash
cd /working2/arctic/project2/SNRAware
python train_multicoil.py \
  --config configs/multicoil/fastmri_x8_cf004_partial05_gmap_ones.yaml \
  --set base_model.variant=small \
  --dry-run
```

## Important Config Fields

- `preprocess.acc_factor`, `center_fraction`, `calib_center_fraction`: current
  fair default is x8 and 0.04/0.04.
- `preprocess.gmap_value`: always `1.0` for the training path.
- `preprocess.cache_dir`: transparent cache for preprocessed arrays and metadata.
- `correction`: bounds for learned complex scale and effective gmap correction.
- `lora`: regex-selected LoRA modules inside the frozen SNRAware base model.
- `train.patch.depth`: defaults to `1` for 2D; set to `16` explicitly for the
  only supported 3D path.
- `subset`: optional training subset, with current public config using 5% random
  volume sampling.
- `base_model.variant`: `large` or `small`; paths are resolved from the local
  ignored `checkpoints/{large,small}` directories unless explicitly overridden.

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
The new folder is training-only for public-dataset fine-tuning.
