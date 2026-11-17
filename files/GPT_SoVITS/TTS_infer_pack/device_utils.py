"""Backend-neutral device helpers shared by the TTS pack and WebUI launchers.

The upstream Intel fork assumes ``torch.xpu`` exists. These helpers keep the
same behavior on Intel XPU while allowing CUDA/ROCm, Apple MPS, DirectML and
plain CPU to work as well.
"""

import torch


def _backend_available(name: str) -> bool:
    name = str(name).strip().lower()
    try:
        if name.startswith("cuda"):
            return torch.cuda.is_available()
        if name.startswith("xpu"):
            return hasattr(torch, "xpu") and torch.xpu.is_available()
        if name.startswith("mps"):
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if name.startswith("dml") or name.startswith("directml"):
            try:
                import torch_directml  # noqa: F401

                return True
            except Exception:
                return False
        if name.startswith("cpu"):
            return True
    except Exception:
        return False
    return False


def _make_device(name: str):
    name = str(name).strip().lower()
    if name.startswith("dml") or name.startswith("directml"):
        import torch_directml

        return torch_directml.device()
    return torch.device(name)


def resolve_device(device=None):
    """Resolve a configured device to an available torch device.

    ``auto`` (or an unavailable explicit device) falls back in this order:
    CUDA/ROCm -> Intel XPU -> Apple MPS -> DirectML -> CPU.
    """
    requested = str(device).strip() if device not in (None, "") else "auto"
    lowered = requested.lower()

    if lowered not in ("auto", "none"):
        if _backend_available(lowered):
            return _make_device(lowered)
        print("Warning: device '%s' is not available, falling back to auto selection." % device)

    for candidate in ("cuda", "xpu", "mps", "dml", "cpu"):
        if _backend_available(candidate):
            return _make_device(candidate)
    return torch.device("cpu")


def empty_device_cache(device=None):
    """Release cached memory on whichever backend the given device uses."""
    name = str(device).strip().lower() if device is not None else "auto"
    try:
        if name.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif name.startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif (
            name.startswith("mps")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            torch.mps.empty_cache()
    except Exception:
        pass
