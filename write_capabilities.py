"""Write BSAI_PATCH.json into a backend root so the proxy can read what this
patch actually supports.

The proxy (bsai/server_intel.py) intersects this declaration with the
backend's own `GPT_SoVITS/configs/tts_infer.yaml` version sections, so a
machine that only has the v4 half of the patch will never advertise v5turbo.

    python write_capabilities.py <backend_root>

Called automatically by apply_patch.sh / apply_patch.ps1 after a successful
apply; it only writes ONE new file (BSAI_PATCH.json) at the backend root and
never touches backend sources.
"""
import json
import os
import sys
from pathlib import Path

# 这个补丁包提供的能力（随补丁内容更新）
PATCH_MODELS = ["v4", "v5turbo"]
PATCH_FEATURES = {
    "batch_synthesize": "/tts_batch",          # 多句一次前向、逐句返回
    "stream_synthesize": "/tts_batch_stream",  # 动态接活 + 逐句流式
    "mmap_weights": True,                      # 权重 mmap 加载，省内存
    "device_agnostic": True,                   # cuda/rocm/xpu/mps/dml/cpu
}


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    if not (root / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py").exists():
        print(f"ERROR: {root} does not look like a GPT-SoVITS-For-Intel checkout",
              file=sys.stderr)
        return 1

    # 只声明后端 tts_infer.yaml 里真的存在的版本段
    present = set()
    cfg = root / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    if cfg.exists():
        try:
            import yaml
            y = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            present = {str(k) for k in y.keys()}
        except Exception:
            present = set()
    models = [m for m in PATCH_MODELS if not present or m in present]

    info = {
        "patch": "BSAIReader GPT-SoVITS-For-Intel patch",
        "models": models,
        "features": PATCH_FEATURES,
        "backend": str(root),
    }
    out = root / "BSAI_PATCH.json"
    out.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}: models={models}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
