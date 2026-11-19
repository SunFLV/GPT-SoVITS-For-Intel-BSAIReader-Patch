"""实例池 worker：独立进程，stdin/stdout 走 JSON 行协议。

由 tts_pool.TTSPool 通过 subprocess 拉起；父进程**不导入 torch**，
避免父进程已初始化 HIP 上下文时子进程加载权重报 hipErrorInvalidImage（ROCm 实测）。

后端中立：
  --device auto|cuda|cuda:N|xpu|mps|dml|cpu   交给 TTS_infer_pack.device_utils 解析
  --gpu N                                     只有显式给出时才设 HIP/CUDA_VISIBLE_DEVICES
  MIOPEN_* 仅在 ROCm 构建下设置

协议（每行一个 JSON）：
  worker -> parent : {"type":"ready","device":"cuda"} | {"type":"warm"} | {"type":"warn","msg":..}
                     {"type":"fatal","msg":..}
                     {"type":"ok","id":N,"sr":48000,"b64":"<int16 PCM base64>"}
                     {"type":"err","id":N,"msg":"..."}
  parent -> worker : {"id":N,"inputs":{...}} | {"type":"shutdown"}
"""

import argparse
import base64
import io
import json
import os
import sys
import tempfile


def _setup_stdio():
    """统一 stdin/stdout 为 UTF-8，避免中文经 cp936 解析导致 UnicodeDecodeError。"""
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )


def main():
    _setup_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="tts_infer.yaml 路径（默认 <项目>/GPT_SoVITS/configs/tts_infer.yaml）")
    ap.add_argument("--device", default="auto", help="auto|cuda|cuda:N|xpu|mps|dml|cpu")
    ap.add_argument("--gpu", type=int, default=None,
                    help="物理 GPU 序号（可选；给了才设可见性环境变量）")
    ap.add_argument("--warm-ref", default=None)
    ap.add_argument("--warm-text", default="预热")
    ap.add_argument("--warm-prompt", default=None)
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    config = args.config or os.path.join(root, "GPT_SoVITS", "configs", "tts_infer.yaml")

    # 可见性环境变量必须在 import torch 之前设置；只有显式指定 --gpu 时才动
    if args.gpu is not None and args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        if sys.platform != "darwin":
            os.environ.setdefault("HIP_VISIBLE_DEVICES", str(args.gpu))

    if root not in sys.path:
        sys.path.insert(0, root)
    gsv = os.path.join(root, "GPT_SoVITS")
    if gsv not in sys.path:
        sys.path.insert(0, gsv)
    if os.getcwd() != root:
        os.chdir(root)  # TTS 内部大量相对路径，统一到项目根

    import warnings

    warnings.filterwarnings("ignore")
    import torch

    # ROCm 下多进程并发加载时 MIOpen 的 kernel DB 会互相踩踏，给每个进程独立目录
    if getattr(torch.version, "hip", None):
        db_dir = os.path.join(tempfile.gettempdir(), f"gsv_miopen_{os.getpid()}")
        os.makedirs(db_dir, exist_ok=True)
        os.environ["MIOPEN_USER_DB_PATH"] = db_dir
        os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = db_dir
        os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

    import numpy as np
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    def emit(obj):
        sys.stdout.write(json.dumps(obj, ensure_ascii=True) + "\n")
        sys.stdout.flush()

    try:
        cfg = TTS_Config(config)
        if args.device and str(args.device).strip().lower() not in ("", "auto"):
            try:
                from TTS_infer_pack.device_utils import resolve_device

                resolved = resolve_device(args.device)
                if str(resolved) != str(cfg.device):
                    print(f"[worker] device {cfg.device} -> {resolved}", flush=True)
                    cfg.device = resolved
            except Exception as exc:
                print(f"[worker] device 解析失败，沿用配置: {exc!r}", flush=True)
        tts = TTS(cfg)
        device = str(getattr(tts.configs, "device", "?"))
    except Exception as exc:
        emit({"type": "fatal", "msg": f"{type(exc).__name__}: {exc}"})
        return 1
    emit({"type": "ready", "device": device, "pid": os.getpid()})

    if args.warm_ref and os.path.exists(args.warm_ref):
        try:
            _generate(tts, {
                "text": args.warm_text, "text_lang": "zh", "ref_audio_path": args.warm_ref,
                "prompt_text": args.warm_prompt or args.warm_text, "prompt_lang": "zh",
                "streaming_mode": 0, "seed": 42,
            })
            emit({"type": "warm", "pid": os.getpid()})
        except Exception as exc:
            emit({"type": "warn", "msg": f"warmup failed: {exc!r}"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except Exception:
            continue
        if job.get("type") == "shutdown":
            break
        jid = job.get("id")
        try:
            sr, audio = _generate(tts, job.get("inputs") or {})
            b64 = base64.b64encode(np.asarray(audio, dtype=np.int16).tobytes()).decode("ascii")
            emit({"type": "ok", "id": jid, "sr": int(sr), "b64": b64})
        except Exception as exc:
            emit({"type": "err", "id": jid, "msg": f"{type(exc).__name__}: {exc}"})
    return 0


def _generate(tts, inputs):
    import numpy as np

    payload = {k: v for k, v in inputs.items() if v is not None}
    chunks = []
    sr = None
    for sr, aud in tts.run(payload):
        chunks.append(np.asarray(aud))
    if not chunks:
        raise RuntimeError("TTS.run 未产出音频")
    audio = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
    return sr, audio


if __name__ == "__main__":
    sys.exit(main())
