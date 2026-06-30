我按你上传的 `SNR_adapter-main.zip` 重新制定。核心方向改成：**直接做一个干净的 3D-only fine-tuning pipeline，不再保留 2D training 兼容分支；但 preprocessing 的物理路径，尤其 image-domain crop→FFT→k-space 操作→noise covariance whitening→coil compression/GRAPPA/SENSE combine，必须原样保留。**

SNRAware 的形状依据仍然是 `[B,C,F,H,W]`，其中 `F` 可以是 depth/slice；输入通道是 real、imag、g-factor，输出是 complex 两通道。  论文也强调 g-factor map 对空间变异噪声很关键，Figure E2 专门展示了 R=2–5 g-factor 噪声放大的鲁棒性。 所以这次修改不能只是把 base model 改成 `D=16`，**g-factor/gmap adapter 也必须一起变成真正支持 `[B,3,16,H,W]` 的 3D 可微模块**。

下面这份可以直接交给 code agent。

---

# Code Agent 计划书：SNR_adapter 改成干净的 3D SNRAware LoRA + gmap adapter pipeline

## 0. 总目标

把当前 multicoil fine-tuning pipeline 从：

```text
当前：
per-slice preprocessing
-> train sample [3,64,64]
-> wrapper unsqueeze 成 [B,3,1,64,64]
-> base model D=1
```

改成：

```text
目标：
per-slice preprocessing 保留不动
-> volume stacking
-> train sample [3,16,64,64]
-> batch [B,3,16,64,64]
-> 3D gmap adapter / correction adapter
-> SNRAware base model D=16
-> output [B,2,16,64,64]
```

推理/验证时：

```text
full volume input:
[1,3,D,384,384]

3D sliding-window:
patch [1,3,16,64,64]

full volume output:
[1,2,D,384,384]
```

`D` 是真实 slice/depth 方向。**不要随机抽 16 张不连续 slices。不要继续逐 slice `[B,3,1,H,W]`。不要为了兼容 2D 留大量分支。**

---

# 1. 不可改动的 preprocessing 边界

当前 `preprocess.py` / `physics.py` 里的这条路径必须保留：

```text
raw multicoil kspace slice
-> ifft2c_np
-> image-domain center crop/pad 到 [384,384]
-> fft2c_np 回到 uniform kspace grid
-> estimate_noise_covariance
-> whitening
-> apply_coil_matrix(kspace, whitening.T)
-> estimate_scc_matrix
-> apply_coil_matrix(whitened, scc)
-> clean SENSE combine
-> undersampling mask
-> GRAPPA
-> noisy SENSE combine
-> per-slice complex scale normalization
-> stored gmap = ones
```

这里 `whitening` 是高风险点，不能“顺手重构”。尤其不要改：

```python
whitening = np.linalg.solve(chol, identity)
whitened = apply_coil_matrix(kspace, whitening.T)
```

也不要改 `apply_coil_matrix` 的 einsum 方向：

```python
np.einsum("chw,cn->nhw", kspace, matrix, optimize=True)
```

执行要求：

```text
[ ] preprocess.py 不做功能性修改。
[ ] physics.py 不做功能性修改。
[ ] crop_size 仍然是 image-domain crop/pad 到 [384,384] 后再 FFT 回 kspace。
[ ] whitening orientation 不变：apply_coil_matrix(kspace, whitening.T)。
[ ] SCC 仍然发生在 whitening 之后。
[ ] GRAPPA、SENSE combine、scale_percentile 逻辑不变。
[ ] stored gmap 仍可为 ones；gmap 的有效估计交给 trainable adapter。
```

如果 code agent 认为必须碰 `preprocess.py` 或 `physics.py`，要停止并说明理由。这个计划里**不做 fixed preprocessed h5 的改造**；未来你会单独做。

---

# 2. 删除 2D 兼容思路，统一 3D shape contract

把 multicoil 项目内部统一成一个形状约定：

```text
Tensor order:
[B, C, D, H, W]

训练 patch:
noisy: [B,3,16,64,64]
clean: [B,2,16,64,64]

验证/推理 full volume:
noisy: [1,3,D,384,384]
clean: [1,2,D,384,384]

SNRAware base config order:
dataset.cutout_shape = [H,W,D] = [64,64,16]

SNRAware model constructor:
D=16, H=64, W=64, C_in=3, C_out=2
```

为了可读性，不要再用裸 list 到处传 `[64,64,16]`。改成命名 dataclass：

```python
@dataclass(frozen=True)
class PatchShape3D:
    depth: int = 16
    height: int = 64
    width: int = 64

    def as_tensor_dhw(self) -> tuple[int, int, int]:
        return (self.depth, self.height, self.width)

    def as_snraware_cutout_hwd(self) -> list[int]:
        return [self.height, self.width, self.depth]
```

同理 overlap：

```python
@dataclass(frozen=True)
class OverlapShape3D:
    depth: int = 8
    height: int = 16
    width: int = 16
```

YAML 改成命名字段，而不是 list：

```yaml
train:
  patch:
    depth: 16
    height: 64
    width: 64
  inference_overlap:
    depth: 8
    height: 16
    width: 16
```

验收：

```text
[ ] 代码里不再出现 train_patch_size: [64,64]。
[ ] multicoil 训练路径不再接受 [B,3,H,W]。
[ ] wrapper.forward 只接受 [B,3,D,H,W]。
[ ] D=16 是模型初始化参数，不是临时 unsqueeze 得到的 singleton 维度。
[ ] 日志第一行明确打印 tensor contract: [B,C,D,H,W]。
```

---

# 3. Volume dataset：保留 per-slice preprocessing，但训练输出 3D window

当前 `h5_dataset.py` 是 `MulticoilH5Dataset`，按 slice 读 H5，然后输出：

```text
noisy: [3,H,W]
clean: [2,H,W]
```

需要改成 3D-only dataset。建议新建或替换为：

```text
src/snraware/projects/mri/multicoil/volume_dataset.py
```

核心类：

```python
@dataclass(frozen=True)
class SliceRef:
    path: Path
    volume_name: str
    slice_idx: int
    source_fingerprint: str

@dataclass(frozen=True)
class VolumeRef:
    volume_name: str
    slices: tuple[SliceRef, ...]
```

数据读取仍然复用当前逻辑：

```text
_read_kspace_slice
_read_target_slice
preprocess_multicoil_slice
cache_path_for_slice / load_preprocess_cache / save_preprocess_cache
```

不要改 preprocessing，只在 dataset 层把 slice stacking 成 volume/window。

## 3.1 Train sample

训练时随机选择一个 volume，再随机选择连续 z-window：

```python
z0 = randint(0, D - patch_depth)
z_indices = [z0, z0+1, ..., z0+15]
```

对这 16 个 slices 逐个调用现有 per-slice preprocessing/cache，然后 stack：

```text
per slice noisy: [3,384,384]
per slice clean: [2,384,384]

stack 后:
noisy_volume_window: [3,16,384,384]
clean_volume_window: [2,16,384,384]
```

然后随机 spatial crop：

```text
top:left crop only on H/W
[3,16,64,64]
[2,16,64,64]
```

metadata 至少保留：

```python
{
    "volume_name": str,
    "z_start": int,
    "z_indices": list[int],
    "patch_top": int,
    "patch_left": int,
    "slice_scales": list[float],  # 每个 slice 的 preprocessing scale
}
```

注意：当前 preprocessing 是 per-slice scale normalization。验证/metric 需要 per-slice scale 还原，所以 `slice_scales` 必须保留。

验收：

```text
[ ] train __getitem__ 返回 noisy.shape == [3,16,64,64]。
[ ] train __getitem__ 返回 clean.shape == [2,16,64,64]。
[ ] collate 后 noisy.shape == [B,3,16,64,64]。
[ ] z_indices 连续。
[ ] z_indices 来自同一个 volume。
[ ] spatial crop 只发生在 H/W，不发生在 D。
[ ] preprocessing/cache 仍然是 per-slice 级别。
```

## 3.2 Validation/test sample

验证和测试不再 per-slice batch，而是每个 item 返回一个 full volume：

```text
noisy: [3,D,384,384]
clean: [2,D,384,384]
metadata:
  volume_name
  z_indices
  slice_scales
```

validation/test loader 建议固定 `batch_size=1`，避免不同 volume 的 D 不一致导致 collate 复杂化。

验收：

```text
[ ] val/test dataloader batch noisy.shape == [1,3,D,384,384]。
[ ] clean.shape == [1,2,D,384,384]。
[ ] metadata["slice_scales"] 长度 == D。
[ ] 不再通过 slice_idx group 回 volume；eval 直接就是 volume-level。
```

---

# 4. G-factor map adapter：改成真正 3D U-Net adapter

当前上传代码里的 `PhysicsCorrectionAdapter` 实际是 2D ConvNet，并且 `_prepare_native_input` 明确拒绝 `T/D != 1`。这个必须重写。

目标模块：

```text
src/snraware/projects/mri/multicoil/gmap_adapter.py
```

建议实现：

```python
class ConvBlock3D(nn.Module):
    ...

class GFactorUNet3D(nn.Module):
    """
    Input:  [B,3,D,H,W]
    Output: [B,1,D,H,W] log_gmap_delta
    """

class GFactorCorrectionAdapter3D(nn.Module):
    """
    Input:  [B,3,D,H,W] = real, imag, initial_gmap
    Output: [B,3,D,H,W] = scaled real/imag, corrected gmap
    """
```

forward 逻辑：

```python
log_scale = complex_log_scale_bound * tanh(log_complex_scale)
complex_scale = exp(log_scale)

log_gmap_delta = gmap_log_bound * tanh(gmap_unet(x))
gmap_ratio = exp(log_gmap_delta)

corrected_complex = x[:, 0:2] * complex_scale
corrected_gmap = clamp(x[:, 2:3] * gmap_ratio, gmap_min, gmap_max)

return cat([corrected_complex, corrected_gmap], dim=1)
```

初始化要求：

```text
[ ] gmap_unet 最后一层 zero-init。
[ ] 初始 log_gmap_delta == 0。
[ ] 初始 gmap_ratio == 1。
[ ] 初始 corrected_gmap == input gmap。
[ ] 初始 complex_scale == 1。
```

这样新 3D adapter 的初始行为等价于当前 ones-gmap，不会一开始破坏已跑通的分布。

U-Net 结构不要过度复杂。推荐 2-level 或 3-level 3D U-Net：

```text
in 3 channels
-> 32
-> 64
-> 128
-> upsample
-> skip concat
-> output 1 channel
```

可以用 `Conv3d + GroupNorm + SiLU`。下采样可以用 stride-2 Conv3d；输入 patch 是 `D=16,H=64,W=64`，下采样两次后是 `4×16×16`，足够稳定。

训练阶段保持当前 warmup 思路，但改名更清楚：

```text
epoch < gmap_warmup_epochs:
  train gmap_unet only
  freeze complex scale
  freeze LoRA/pre/post

gmap_warmup_epochs <= epoch < warmup_epochs:
  train gmap_unet + complex scale
  freeze LoRA/pre/post

epoch >= warmup_epochs:
  train gmap_unet + complex scale + LoRA + optional pre/post
```

验收：

```text
[ ] adapter.forward 只接受 [B,3,D,H,W]。
[ ] adapter 输出 [B,3,D,H,W]。
[ ] 不存在 x.ndim == 4 分支。
[ ] 不存在 Expected singleton T=1。
[ ] zero-init 时输出第三通道与输入第三通道 allclose。
[ ] gmap_unet 参数在 gmap warmup 阶段 requires_grad=True。
[ ] LoRA 参数在 joint 阶段 requires_grad=True。
```

---

# 5. SNRAware wrapper：直接 D=16，不再 unsqueeze

重写 `snraware_wrapper.py` 中的 shape 部分。

当前错误点：

```python
fixed.dataset.cutout_shape = [H, W, 1]
DenoisingModel(... D=1 ...)
base_input = x.unsqueeze(2)
return y.squeeze(2)
```

目标：

```python
fixed.dataset.cutout_shape = patch_shape.as_snraware_cutout_hwd()
model = DenoisingModel(
    config=model_config,
    D=patch.depth,
    H=patch.height,
    W=patch.width,
    C_in=3,
    C_out=2,
)
```

wrapper forward：

```python
def forward(self, x: torch.Tensor, *, checkpoint_base_model: bool = False) -> torch.Tensor:
    assert x.shape[1:] == [3,16,64,64] for training patch path
    x = self.gmap_adapter(x)
    y = self.base_model(x)
    assert y.shape[1:] == [2,16,64,64]
    return y
```

验证 full volume 不直接进 wrapper；它先由 sliding-window 拆成 `[Bpatch,3,16,64,64]` 再进 wrapper。

验收：

```text
[ ] model_config.dataset.cutout_shape == [64,64,16]。
[ ] DenoisingModel 初始化参数 D=16, H=64, W=64。
[ ] random tensor [1,3,16,64,64] forward 输出 [1,2,16,64,64]。
[ ] 没有 unsqueeze/squeeze singleton D 的代码。
[ ] base checkpoint strict load 成功；若失败，错误信息必须列出 mismatched keys。
```

---

# 6. 3D sliding-window inference/eval

当前 trainer 里的 `_predict_sliding_window_eval` 是 2D sliding-window，只处理 `[B,C,H,W]`。需要重写为 3D。

输入：

```text
noisy: [1,3,D,384,384]
```

patch shape：

```text
patch_d = 16
patch_h = 64
patch_w = 64
```

overlap：

```text
overlap_d = 8
overlap_h = 16
overlap_w = 16
```

stride：

```text
stride_d = 8
stride_h = 48
stride_w = 48
```

positions：

```python
z_positions = patch_positions(D, 16, 8)
y_positions = patch_positions(H, 64, 16)
x_positions = patch_positions(W, 64, 16)
```

每个 patch：

```text
patch = noisy[:, :, z:z+16, y:y+64, x:x+64]
pred_patch = model(patch)
```

stitch 方式不要只做 count-average。做 separable ramp blending：

```text
weight_d: [16]
weight_h: [64]
weight_w: [64]

weight = weight_d[:,None,None] * weight_h[None,:,None] * weight_w[None,None,:]
weight shape -> [1,1,16,64,64]
```

累计：

```python
prediction_sum[:, :, z:z+16, y:y+64, x:x+64] += pred_patch * weight
weight_sum[:, :, z:z+16, y:y+64, x:x+64] += weight
output = prediction_sum / weight_sum.clamp_min(eps)
```

验收：

```text
[ ] full input [1,3,D,384,384] 输出 [1,2,D,384,384]。
[ ] weight_sum.min() > 0。
[ ] identity dummy model 的 sliding-window 输出与 direct output allclose。
[ ] eval_patch_batch_size 控制同时 forward 的 patch 数，避免 OOM。
[ ] 384×384 不 resize；只 sliding crop/stitch。
```

---

# 7. Loss、metrics、scale restoration 全部 5D

重写以下函数，让它们只处理 5D：

```text
complex_magnitude
fastmri_current_magnitude_mean
_loss
_metadata_to_numpy / metric preparation
```

目标：

```python
def complex_magnitude(x):
    # x: [B,2,D,H,W]
    return torch.sqrt(x[:,0:1].square() + x[:,1:2].square())

def current_magnitude_mean(noisy):
    # noisy: [B,3,D,H,W]
    mag = sqrt(real^2 + imag^2)
    scale = mag.mean(dim=(-3,-2,-1), keepdim=True)
    return scale.clamp_min(...)
```

validation metric 的关键点：preprocessing 现在仍是 per-slice scale，所以 full volume metric 不能只乘一个 scalar。dataset metadata 需要提供：

```python
slice_scales: list[float]  # length D
```

还原 magnitude：

```python
pred_mag:   [D,H,W]
target_mag: [D,H,W]
scales:     [D]

pred_restore[z] = pred_mag[z] * scales[z]
target_restore[z] = target_mag[z] * scales[z]
```

验收：

```text
[ ] train loss 输入 pred/clean/noisy 均为 5D。
[ ] scale.shape == [B,1,1,1,1]。
[ ] val metric 使用 per-slice scales 还原。
[ ] metric 直接基于 volume [D,H,W] 计算，不再 group per-slice predictions。
```

---

# 8. Trainer：改成 3D-only，删除 2D 分支

`trainer.py` 需要清理成单一路径：

```text
train_epoch:
  batch["noisy"] [B,3,16,64,64]
  batch["clean"] [B,2,16,64,64]
  pred = model(noisy)
  loss = loss_5d(pred, clean, noisy)

evaluate:
  batch["noisy"] [1,3,D,384,384]
  batch["clean"] [1,2,D,384,384]
  pred = predict_sliding_window_3d(noisy)
  loss = loss_5d(pred, clean, noisy)
  metrics = volume_metrics(pred, clean, slice_scales)
```

删除或替换以下 2D 逻辑：

```text
[B,3,H,W] checks
base_input = unsqueeze(2)
squeeze(2)
_group_slices_into_volumes
per-slice metadata_to_numpy
2D sliding-window eval
legacy [64,64] config validation
```

保留 LoRA 注入、optimizer group、warmup schedule、checkpoint 保存思路，但 checkpoint 类型更新：

```python
MULTICOIL_CHECKPOINT_TYPE = "snraware_multicoil_3d_v1"
```

checkpoint 内容：

```python
{
  "checkpoint_type": "snraware_multicoil_3d_v1",
  "shape_contract": "[B,C,D,H,W]",
  "patch": {"depth":16,"height":64,"width":64},
  "gmap_adapter": ...,
  "lora_adapter": ...,
  "optimizer_state_dict": ...,
  "config": ...
}
```

不需要兼容旧 2D checkpoint。旧 checkpoint resume 时直接失败，错误信息明确写：

```text
This is a 3D-only pipeline. 2D adapter checkpoints are not supported.
```

验收：

```text
[ ] trainer 中没有 4D training path。
[ ] trainer 中没有 singleton D path。
[ ] checkpoint type 是 3D v1。
[ ] 旧 checkpoint 不静默加载。
[ ] metrics.csv 记录 gmap_mean/gmap_p95/gmap_max/complex_scale。
```

---

# 9. Config：新建干净 3D YAML，旧 YAML 可直接替换或弃用

因为你不再做 2D training，不需要维护旧 YAML。建议直接把 multicoil config 改成 3D：

```yaml
preprocess:
  crop_size: [384, 384]
  acc_factor: 8
  center_fraction: 0.04
  calib_center_fraction: 0.04
  sampling_pattern: uniform
  ncc: 8
  grappa_kernel: [5, 5]
  grappa_lambda: 0.0001
  cov_corner_fraction: 0.125
  cov_shrinkage: 0.05
  cov_condition_max: 1000000.0
  eig_floor: 0.000001
  scale_percentile: 50.0
  deterministic_mask_from_name: true
  sample_seed: 42
  gmap_value: 1.0
  cache_dir: /working2/arctic/project2/cache/fastmri_x8_cf004_gmap_ones
  cache_version: multicoil_ones_gmap_v1

correction:
  enabled: true
  hidden_chans: 32
  gmap_log_bound: 1.75
  complex_log_scale_bound: 0.75
  gmap_min: 0.01
  gmap_max: 12.0

train:
  max_epochs: 50
  warmup_epochs: 4
  gmap_warmup_epochs: 2
  batch_size: 2
  val_batch_size: 1
  num_workers: 4
  patch:
    depth: 16
    height: 64
    width: 64
  inference_overlap:
    depth: 8
    height: 16
    width: 16
  eval_patch_batch_size: 8
  gradient_checkpoint_frozen_base: true
  frozen_base_eval: true
  correction_lr: 0.0005
  adapter_lr: 0.0001
  train_pre_post: true
```

`batch_size` 不要沿用 48。3D patch 体素数是 2D patch 的 16 倍，先从 1–2 起步。

验收：

```text
[ ] YAML 里不存在 train_patch_size: [64,64]。
[ ] YAML 里不存在 overlap_for_inference: [16,16,0]。
[ ] run_name 明确包含 3d_d16 或类似字段。
[ ] 启动日志打印 patch D/H/W 与 full crop H/W。
```

---

# 10. 文件组织建议

为了人类可读性，不要把所有东西堆在一个 `adapter.py` 里。建议：

```text
src/snraware/projects/mri/multicoil/
  config.py              # typed config, named 3D patch/overlap
  preprocess.py          # untouched
  physics.py             # untouched
  cache.py               # mostly unchanged
  volume_dataset.py      # VolumeRef, per-slice preprocess -> 3D windows/full volume
  gmap_adapter.py        # GFactorUNet3D + GFactorCorrectionAdapter3D
  lora.py                # LoRA wrappers/injection, 从旧 adapter.py 拆出
  snraware_wrapper.py    # 3D-only base wrapper
  sliding_window.py      # patch positions + 3D ramp stitch
  metrics.py             # 5D magnitude, PSNR/SSIM/NMSE, scale restoration
  trainer.py             # 3D trainer
```

不要添加：

```text
*_old.py
*_backup.py
legacy_2d.py
兼容 4D 的 if/else
大段注释掉的旧代码
临时 smoke scripts
```

测试可以保留在 `test/`，但不要在 src 里塞调试脚本。

---

# 11. 必须有的最小 tests

不要写大量无关 smoke tests，只写 shape/contract tests。

```text
test_multicoil_preprocess_regression.py
test_volume_dataset_3d_shapes.py
test_gmap_adapter_3d.py
test_snraware_wrapper_3d_shape.py
test_sliding_window_3d_identity.py
test_loss_metrics_3d.py
```

## 11.1 preprocessing regression

目的：证明 whitening/crop/FFT/kspace path 没被改。

做法：

```text
1. 固定 random seed 生成 synthetic complex kspace [coil,H,W]。
2. 调用 preprocess_multicoil_slice。
3. 保存或在测试内比较关键 contract：
   - output shapes
   - finite
   - metadata whiten_mode/cov_condition 存在
   - gmap 全 ones
   - scale finite
4. 如果 PR 里 preprocess.py/physics.py 被修改，必须增加 before/after numeric allclose 说明。
```

更强要求：code agent 在 PR 描述中明确写：

```text
preprocess.py diff: none
physics.py diff: none
```

## 11.2 dataset shapes

用 monkeypatch 的 `_load_or_preprocess` 返回 fake slice result，不依赖真实 H5/GRAPPA。

验收：

```text
train sample noisy [3,16,64,64]
train sample clean [2,16,64,64]
z_indices 连续
slice_scales length 16
val sample noisy [3,D,384,384]
```

## 11.3 gmap adapter

验收：

```text
x = torch.randn(2,3,16,64,64)
x[:,2] = 1

adapter zero-init:
out.shape == x.shape
out[:,2] allclose 1
out[:,0:2] allclose x[:,0:2]
loss.backward 后 gmap_unet 有梯度
```

## 11.4 sliding-window identity

用 dummy model：

```python
class Dummy(nn.Module):
    def forward(self, x):
        return x[:, 0:2]
```

输入 full volume：

```text
[1,3,D,384,384]
```

sliding-window 输出应 allclose `input[:,0:2]`。

---

# 12. Code agent 执行顺序

按这个顺序做，不要并行大改：

## Step 1：写 shape config

```text
[ ] PatchShape3D
[ ] OverlapShape3D
[ ] ProjectConfig YAML parsing
[ ] 删除 2D train_patch_size list 校验
```

完成后先跑 config 单测。

## Step 2：volume dataset

```text
[ ] VolumeRef grouping
[ ] train contiguous D-window
[ ] full val/test volume
[ ] per-slice cache 复用
[ ] metadata slice_scales
```

完成后只跑 dataset shape tests。

## Step 3：gmap adapter 3D

```text
[ ] GFactorUNet3D
[ ] GFactorCorrectionAdapter3D
[ ] zero-init identity behavior
[ ] last_stats 支持 5D
```

完成后跑 adapter tests。

## Step 4：wrapper D=16

```text
[ ] base config cutout_shape [64,64,16]
[ ] DenoisingModel D=16
[ ] wrapper forward 5D-only
[ ] strict base checkpoint load report
```

完成后跑 wrapper shape test。如果真实 SNRAware checkpoint 在环境可用，额外跑一次真实 forward。

## Step 5：trainer/loss/metrics/sliding-window

```text
[ ] 5D loss
[ ] 3D sliding-window
[ ] volume metric scale restoration
[ ] train/eval loop shape update
```

完成后跑 identity stitch 和 loss/metrics tests。

## Step 6：checkpoint + final config

```text
[ ] checkpoint_type 改成 3D v1
[ ] 保存 gmap_adapter + LoRA/pre/post
[ ] 不兼容旧 2D checkpoint
[ ] YAML 改成 3D config
```

完成后做一个最小真实 dry-run：

```text
limit_train_batches=2
limit_val_batches=1
batch_size=1 or 2
```

---

# 13. 最终逐条复核清单

code agent 提交前必须逐条确认：

```text
[ ] preprocess.py 没有功能性修改。
[ ] physics.py 没有功能性修改。
[ ] image-domain crop 到 384×384 后再 FFT 回 kspace 的逻辑保留。
[ ] whitening 仍然是 np.linalg.solve(chol, identity)。
[ ] whitening 应用仍然是 apply_coil_matrix(kspace, whitening.T)。
[ ] SCC 仍然在 whitening 后执行。
[ ] stored preprocessing gmap 仍为 ones/gmap_value，不在 preprocessing 里动态估计。
[ ] 没有实现 fixed preprocessed h5 保存逻辑。
[ ] train tensor 是 [B,3,16,64,64]。
[ ] clean tensor 是 [B,2,16,64,64]。
[ ] validation tensor 是 [1,3,D,384,384]。
[ ] wrapper 不接受 [B,3,H,W]。
[ ] wrapper 没有 unsqueeze(2)/squeeze(2) singleton D 逻辑。
[ ] base model 初始化 D=16,H=64,W=64。
[ ] model_config.dataset.cutout_shape == [64,64,16]。
[ ] gmap adapter 是 3D module，输入输出都是 [B,3,D,H,W]。
[ ] gmap adapter final layer zero-init。
[ ] 初始 gmap_ratio == 1。
[ ] z-window 是连续 slices。
[ ] 没有随机抽 16 个不连续 slices。
[ ] eval 是 3D sliding-window，不是 2D H/W sliding-window。
[ ] stitching 使用 ramp weight，不是简单硬拼接。
[ ] volume metrics 使用 per-slice scale restoration。
[ ] checkpoint type 是 3D v1。
[ ] 旧 2D checkpoint 不被静默加载。
[ ] 代码库没有 *_old.py、*_backup.py、legacy 4D 分支、注释掉的大段旧代码。
[ ] 第一批 train batch 的 noisy/clean/pred shape 会被清楚打印到日志。
```

---

# 14. 明确不做的事

这次不要做：

```text
1. 不把 preprocessing 改成固定 h5 预计算。
2. 不重写 whitening。
3. 不改 GRAPPA/SENSE/crop/scale 的物理路径。
4. 不保留 2D training compatibility。
5. 不写 4D/5D 双路径。
6. 不加大量 fallback/outlier handling。
7. 不做 D<16 的复杂 padding 兜底；当前假设 preprocessing/数据准备保证 D 足够且 H/W 是目标尺寸。
```

这版修改完成后，代码应该只表达一件事：**fastMRI multicoil slice-level physics preprocessing 保持原样；训练和推理统一进入 SNRAware-compatible 的 3D `[B,3,D,H,W]` pipeline；D-window 固定 16；gmap adapter 和 LoRA 一起在 3D patch 上微调。**
