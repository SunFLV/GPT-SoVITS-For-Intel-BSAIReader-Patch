"""把 s2Gv5turbo.pth 的 Linear 权重量化成 INT8（按输出通道缩放）。

只改 ``nn.Linear`` 的二维权重；embedding / conv / 归一化层保持 fp16。
生成 ``s2Gv5turbo_int8.pth``，配合 ``bsai_int8.py`` 使用。

用法：
    runtime_cuda/python.exe quantize_v5turbo_int8.py
    runtime_cuda/python.exe quantize_v5turbo_int8.py --verify
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch import nn

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "GPT_SoVITS", "pretrained_models", "gsv-v5-pretrained", "s2Gv5turbo.pth")
DST = os.path.join(ROOT, "GPT_SoVITS", "pretrained_models", "gsv-v5-pretrained", "s2Gv5turbo_int8.pth")


def _linear_weight_keys(ckpt):
    """用 checkpoint 里的 config 构造一次模型，拿到所有 nn.Linear 权重键。"""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    gsv = os.path.join(ROOT, "GPT_SoVITS")
    if gsv not in sys.path:
        sys.path.insert(0, gsv)
    from GPT_SoVITS.module.models import SynthesizerTrnV3

    hps = dict(ckpt["config"])
    hps["model"] = dict(hps["model"])
    hps["data"] = dict(hps["data"])
    hps["train"] = dict(hps["train"])
    hps["model"]["version"] = "v5turbo"
    hps["model"]["semantic_frame_rate"] = "25hz"
    model = SynthesizerTrnV3(
        hps["data"]["filter_length"] // 2 + 1,
        hps["train"]["segment_size"] // hps["data"]["hop_length"],
        n_speakers=hps["data"].get("n_speakers", 0),
        **hps["model"],
    )
    return {"%s.weight" % name for name, module in model.named_modules() if isinstance(module, nn.Linear)}


def quantize(src=SRC, dst=DST, verify=False):
    ckpt = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    weights = ckpt["weight"]
    linear_keys = _linear_weight_keys(ckpt)

    out = {}
    quantized = 0
    kept = 0
    for key, value in weights.items():
        if key in linear_keys and getattr(value, "ndim", 0) == 2:
            w = value.detach().to(torch.float32)
            # 对称量化：每行（输出通道）一个 scale，范围 [-127, 127]。
            amax = w.abs().amax(dim=1).clamp_min(1e-8)
            scale = (amax / 127.0).to(torch.float16)
            q = torch.round(w / scale.to(torch.float32)[:, None]).clamp_(-127, 127).to(torch.int8)
            out[key + "_int8"] = q
            out[key.replace(".weight", ".weight_scale")] = scale[:, None].contiguous()
            quantized += 1
        else:
            out[key] = value
            kept += 1

    new_ckpt = dict(ckpt)
    new_ckpt["weight"] = out
    new_ckpt["bsai_int8"] = {
        "scheme": "per_output_channel_symmetric",
        "dtype": "int8",
        "scale_dtype": "float16",
        "quantized_linear_weights": quantized,
        "kept_tensors": kept,
        "source": os.path.basename(src),
    }
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(new_ckpt, dst)
    print("quantized %d Linear weights, kept %d tensors -> %s" % (quantized, kept, dst))
    print("size: %.1f MiB -> %.1f MiB" % (
        os.path.getsize(src) / 1024 ** 2,
        os.path.getsize(dst) / 1024 ** 2,
    ))

    if verify:
        bad = [k for k in linear_keys if k + "_int8" not in out]
        if bad:
            raise RuntimeError("missing quantized keys: %s" % bad[:5])
        # 误差粗查：反量化后的相对 L2，很快。
        errs = []
        for key in list(linear_keys)[:16]:
            w = weights[key].to(torch.float32)
            q = out[key + "_int8"].to(torch.float32)
            s = out[key.replace(".weight", ".weight_scale")].to(torch.float32)
            rec = q * s
            errs.append(((w - rec).norm() / w.norm().clamp_min(1e-8)).item())
        print("relative L2 (first 16 linear layers): mean=%.5f max=%.5f" % (
            sum(errs) / len(errs), max(errs)
        ))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    quantize(args.src, args.dst, args.verify)


if __name__ == "__main__":
    main()
