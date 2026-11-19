"""安装补丁后，为这台后端生成两份能力声明（都是新增文件，不改后端源码）：

    BSAI_PATCH.json    补丁提供的能力（模型版本 / 接口特性）
    BSAI_VOICES.json   这台机器可用的参考音色（名称 / 描述 / 相对路径 / prompt）

后端自己会读它们（`bsai_capabilities.py` 挂载 `/api/voices`、`/api/models`），
bsai 代理层也会读 —— 于是「这台支持什么模型、有哪些音色」不需要额外的代理层
就能被前端问到。

    python write_capabilities.py <后端根目录>
    python write_capabilities.py <后端根目录> --voices-json D:/x/voices.json
    python write_capabilities.py <后端根目录> --name "9070XT"

音色来源优先级：
  1. --voices-json 指定的清单（bsai 代理层的配置也能直接用，字段会自动对齐）
  2. 后端根目录已有的 BSAI_VOICES.json（保留人工维护的内容）
  3. bsai/config_intel.yaml 或 config_intel.yaml 里的 voices:
  4. 扫描 voices/ ref_audios/ SunFLV/Reference_Voices/ 下的音频文件
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

# 这个补丁包提供的能力（随补丁内容更新）
PATCH_MODELS = ["v4", "v5turbo"]
PATCH_FEATURES = {
    "batch_synthesize": "/tts_batch",          # 多句一次前向、逐句返回
    "stream_synthesize": "/tts_batch_stream",  # 动态接活 + 逐句流式
    "capabilities": "/api/capabilities",       # 自报模型/音色
    "voices": "/api/voices",
    "mmap_weights": True,                      # 权重 mmap 加载，省内存
    "device_agnostic": True,                   # cuda/rocm/xpu/mps/dml/cpu
}
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
VOICE_SEARCH_DIRS = ("voices", "ref_audios", "SunFLV/Reference_Voices",
                     "SunFLV/Baked_Voices")


def _present_versions(root: Path) -> set:
    cfg = root / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    if not cfg.exists():
        return set()
    try:
        import yaml
        return {str(k) for k in (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {})}
    except Exception:
        out = set()
        try:
            for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
                if line and not line[0].isspace() and line.rstrip().endswith(":"):
                    out.add(line.strip().rstrip(":"))
        except Exception:
            pass
        return out


def _rel(root: Path, p) -> str:
    s = str(p or "").replace("\\", "/")
    if not s:
        return ""
    try:
        return str(Path(s).resolve().relative_to(root)).replace("\\", "/")
    except Exception:
        pass
    # 路径在后端根目录之外（比如从别的机器抄来的配置）：在根目录里找同名副本
    # （附加包自带的 bsai/voices/ 就是这种情况），找到就写成相对路径 —— 这样
    # 清单换台机器也能用。
    name = Path(s).name
    if name:
        for sub in ("bsai", "voices", "ref_audios"):
            base = root / sub
            if not base.exists():
                continue
            for hit in base.rglob(name):
                if hit.is_file():
                    return str(hit.relative_to(root)).replace("\\", "/")
    return s


def _norm(root: Path, v: dict) -> dict:
    name = v.get("name") or Path(str(v.get("ref_audio_path", "voice"))).stem
    return {
        "name": name,
        "description": v.get("description", ""),
        "text_lang": v.get("text_lang", "zh"),
        "prompt_lang": v.get("prompt_lang", "ja"),
        "ref_audio_path": _rel(root, v.get("ref_audio_path")),
        "prompt_text": v.get("prompt_text", ""),
    }


def _from_bsai_config(root: Path) -> list:
    """bsai 代理层的配置里已经有完整的音色定义（含 prompt_text），直接复用。"""
    for cand in (root / "bsai" / "config_intel.yaml", root / "config_intel.yaml",
                 root / "bsai" / "config.addon.example.yaml"):
        if not cand.exists():
            continue
        try:
            import yaml
            data = yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        vs = data.get("voices")
        if isinstance(vs, list) and vs:
            return [_norm(root, v) for v in vs if isinstance(v, dict)]
    return []


def _scan(root: Path) -> list:
    out, seen = [], set()
    for d in VOICE_SEARCH_DIRS:
        base = root / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS or p in seen:
                continue
            seen.add(p)
            prompt = ""
            side = p.with_suffix(".txt")
            if side.exists():
                try:
                    prompt = side.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
            out.append({
                "name": _rel(root, p).rsplit(".", 1)[0].replace("/", "_"),
                "description": p.parent.name if p.parent != base else "",
                "text_lang": "zh",
                "prompt_lang": "ja" if prompt else "zh",
                "ref_audio_path": _rel(root, p),
                "prompt_text": prompt,
            })
    return out


def _load_any(path: Path):
    """清单既可以是 JSON，也可以是 bsai 的 YAML 配置（直接复用它的 voices:）。"""
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        import yaml
        return yaml.safe_load(text)
    except Exception:
        return None


def collect_voices(root: Path, voices_json: Path | None) -> list:
    if voices_json and voices_json.exists():
        try:
            data = _load_any(voices_json)
            items = data.get("voices") if isinstance(data, dict) else data
            if isinstance(items, list) and items:
                return [_norm(root, v) for v in items if isinstance(v, dict)]
        except Exception as e:
            print(f"WARN: 读不了 {voices_json}: {e}", file=sys.stderr)

    for cand in (root / "BSAI_VOICES.json", root / "bsai" / "BSAI_VOICES.json"):
        if cand.exists():
            try:
                items = json.loads(cand.read_text(encoding="utf-8")).get("voices") or []
                if items:
                    return [_norm(root, v) for v in items if isinstance(v, dict)]
            except Exception:
                pass

    got = _from_bsai_config(root)
    return got or _scan(root)


def main() -> int:
    ap = argparse.ArgumentParser(description="为后端生成 BSAI 能力声明")
    ap.add_argument("root", nargs="?", default=".", help="后端根目录")
    ap.add_argument("--voices-json", default=None, help="外部音色清单（bsai 配置也可）")
    ap.add_argument("--name", default=None, help="这台机器的显示名，如 9070XT")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not (root / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py").exists():
        print(f"ERROR: {root} does not look like a GPT-SoVITS-For-Intel checkout",
              file=sys.stderr)
        return 1

    present = _present_versions(root)
    models = [m for m in PATCH_MODELS if not present or m in present]

    info = {
        "patch": "BSAIReader GPT-SoVITS-For-Intel patch",
        "models": models,
        "features": PATCH_FEATURES,
        # 前端「服务器选择」直接显示这个名字：优先命令行，其次环境变量，
        # 再次机器名 —— 多台机器摆在一起时一眼能分清
        "server_name": (args.name or os.environ.get("BSAI_SERVER_NAME")
                        or socket.gethostname() or root.name),
        "backend": str(root),
    }
    out = root / "BSAI_PATCH.json"
    out.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    vs = collect_voices(root, Path(args.voices_json) if args.voices_json else None)
    vout = root / "BSAI_VOICES.json"
    vout.write_text(json.dumps({"voices": vs}, ensure_ascii=False, indent=2),
                    encoding="utf-8")

    print(f"wrote {out}: models={models}")
    print(f"wrote {vout}: voices={len(vs)}" + (f" ({', '.join(v['name'] for v in vs[:3])}…)" if vs else ""))
    if not vs:
        print("  提示：没找到参考音，可用 --voices-json 指定（bsai 的 config_intel.yaml 也行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
