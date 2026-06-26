# zero_shot

Private or custom H5 inference lives here so it does not leak into the training
package. The entrypoint reuses the multicoil preprocessing schema and exports
per-slice complex predictions as compressed `.npz` files.

This folder is for inference only. Training code stays in
`src/snraware/projects/mri/multicoil/` and `train_multicoil.py`.
