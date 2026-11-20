"""v5Turbo INT8 权重加载插件（仅给 NVIDIA / P106 池服务使用）。

思路
----
预训练的 s2Gv5turbo.pth 里权重已经是 fp16（730 MiB），其中约 444 MiB 是
``nn.Linear`` 的二维矩阵。P106-90 只有 3 GiB 显存，而且 Pascal 没有 Tensor
Core；把 Linear 权重离线转成 INT8，推理时再按输出通道反量化成当前激活 dtype，
可以把常驻权重压到约 365 MiB，同时保持原来的计算路径（fp16 GEMM）。

这不是 FBGEMM/CUDA INT8 GEMM —— Pascal 上那条路要么不可用，要么比 fp16 更慢。
这里做的是「权重量化省显存」，速度收益主要来自避免显存溢出/换页。

用法
----
在导入 ``TTS`` 之前调用 :func:`install`，或直接让 ``tts_worker.py`` 在
``BSAI_INT8_WEIGHTS=1`` 时自动加载本模块。插件只拦截 SoVITS 权重的加载过程，
不影响 GPT/BERT/vocoder。
"""

import os
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

INT8_MARK = "int8"


class Int8Linear(nn.Module):
    """nn.Linear 的权重 INT8 存储版本；接口与 nn.Linear 的推理用法兼容。"""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        factory = {}
        if device is not None:
            factory["device"] = device
        if dtype is not None:
            factory["dtype"] = dtype
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        # 不保留 fp16 的 self.weight，否则省显存的意义就没了。
        self.register_buffer(
            "weight_int8",
            torch.zeros((self.out_features, self.in_features), dtype=torch.int8, **factory),
        )
        scale_dtype = dtype if dtype is not None else torch.float16
        self.register_buffer(
            "weight_scale",
            torch.ones((self.out_features, 1), dtype=scale_dtype, **factory),
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, **factory))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self):
        """兼容直接访问 ``module.weight`` 的推理代码（如 DiT static cache）。

        这里没有常驻 fp16 权重；只在该代码确实需要时临时反量化一份。
        """
        return self.weight_int8.to(dtype=self.weight_scale.dtype) * self.weight_scale

    def forward(self, x):
        # 反量化到激活 dtype；只临时生成当前层的 fp16 权重，不是整模型。
        w = self.weight_int8.to(dtype=x.dtype) * self.weight_scale.to(dtype=x.dtype)
        b = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, w, b)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, weight=int8+fp16_scale".format(
            self.in_features, self.out_features, self.bias is not None
        )


@contextmanager
def int8_linear_scope():
    """只在 SoVITS 模型构造期间把 nn.Linear 换成 Int8Linear。"""
    original = nn.Linear
    nn.Linear = Int8Linear
    try:
        yield
    finally:
        nn.Linear = original


def _is_int8_checkpoint(path):
    return INT8_MARK in os.path.basename(str(path)).lower() or os.environ.get("BSAI_INT8_WEIGHTS") == "1"


def install():
    """给 TTS 打补丁；重复调用安全。"""
    from GPT_SoVITS.TTS_infer_pack import TTS as tts_mod
    from GPT_SoVITS import process_ckpt as ckpt_mod

    if getattr(tts_mod, "_bsai_int8_installed", False):
        return tts_mod

    # 1) 版本识别：量化后的 pth 头 8 KiB 哈希变了，按文件名识别为 v5turbo。
    orig_fast = ckpt_mod.get_sovits_version_from_path_fast

    def get_version_fast(path):
        if INT8_MARK in os.path.basename(str(path)).lower():
            return ("v2", "v5turbo", False)
        return orig_fast(path)

    ckpt_mod.get_sovits_version_from_path_fast = get_version_fast
    # TTS.py 是 from ... import ...，模块里的引用也要一起换。
    tts_mod.get_sovits_version_from_path_fast = get_version_fast

    # 2) 只在加载 int8 权重时替换 Linear，避免影响 BERT / GPT / vocoder。
    orig_init_vits = tts_mod.TTS.init_vits_weights

    def init_vits_weights(self, weights_path):
        if _is_int8_checkpoint(weights_path):
            with int8_linear_scope():
                return orig_init_vits(self, weights_path)
        return orig_init_vits(self, weights_path)

    tts_mod.TTS.init_vits_weights = init_vits_weights
    tts_mod._bsai_int8_installed = True
    return tts_mod
