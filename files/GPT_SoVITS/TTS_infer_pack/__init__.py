# Keep package import lightweight so device_utils can be imported without
# pulling in torch/IPEX-dependent modules. Submodules are imported on demand.
