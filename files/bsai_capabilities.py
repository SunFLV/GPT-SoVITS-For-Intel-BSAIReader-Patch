"""BSAI capability manifest — 每个装了本补丁的后端都能「自报家门」。

补丁装好后，后端根目录会多出两个纯数据文件（由 write_capabilities.py 生成）：

    BSAI_PATCH.json     本补丁提供的能力：支持哪些模型版本、有哪些接口
    BSAI_VOICES.json    这台机器有哪些参考音色（名称/描述/参考音频/prompt）

本模块负责把它们和对端真实的 `GPT_SoVITS/configs/tts_infer.yaml` 交叉校验，
然后以统一的 JSON 形状暴露出来 —— **前端、代理层、池服务看到的都是同一套字段**：

    GET /api/capabilities   完整能力清单
    GET /api/voices         音色 + 模型版本（与 bsai 代理层返回一致，可直接互换）
    GET /api/models         仅模型版本

之所以放进补丁而不是只写代理层：一台机器可能只跑 `/tts_batch` 池服务
（没有 bsai 代理、没有音色列表接口），但它照样知道自己的模型和音色。
挂上这里的路由，前端连过去就能读，不必为了「看有哪些音色」再部署一整套前端。

用法（在 api_v2.py / api_pool.py 里各加三行）：

    try:
        import bsai_capabilities as _caps
        _caps.attach(APP)
    except Exception:
        pass          # 没装补丁的老后端：静默跳过，不影响原有功能

不依赖 yaml / fastapi 之外的任何库；yaml 缺失时退化为简单文本解析。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PATCH_JSON = "BSAI_PATCH.json"
VOICES_JSON = "BSAI_VOICES.json"
TTS_INFER = Path("GPT_SoVITS") / "configs" / "tts_infer.yaml"

# 只有这些版本是「能拿来合成」的；v5dev 是实验分支，不对外宣称
USABLE_VERSIONS = ("v4", "v5turbo")
MODEL_LABELS = {
    "v4": "v4（稳定）",
    "v5turbo": "v5Turbo（4 步快速）",
    "v5dev": "v5dev（实验）",
    "v3": "v3",
    "v2": "v2",
    "v1": "v1",
}
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

# 没写清单时按这些目录找参考音（相对后端根目录）
VOICE_SEARCH_DIRS = ("voices", "ref_audios", "SunFLV/Reference_Voices",
                     "SunFLV/Baked_Voices")


# ---------------------------------------------------------------- 基础读取
def _read_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_versions() -> list:
    """后端 tts_infer.yaml 里真实存在的版本段（按 USABLE_VERSIONS 过滤）。"""
    cfg = ROOT / TTS_INFER
    if not cfg.exists():
        return []
    keys = []
    try:
        import yaml  # 后端自带，正常都装得到
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        keys = [str(k) for k in data.keys()]
    except Exception:
        # 没有 yaml：退化成「行首 xxx:」的粗糙解析，够用
        try:
            for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
                if line and not line[0].isspace() and line.rstrip().endswith(":"):
                    keys.append(line.strip().rstrip(":"))
        except Exception:
            return []
    return [k for k in keys if k in USABLE_VERSIONS]


def _rel(p) -> str:
    """绝对路径 → 相对后端根目录（跨机器部署时路径才有意义）。"""
    if not p:
        return ""
    s = str(p).replace("\\", "/")
    try:
        return str(Path(s).resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return s


def _abs(p) -> str:
    """清单里的相对路径 → 本机绝对路径。已经是绝对路径就原样返回。

    清单可能是从另一台机器带过来的（绝对路径指着那边的盘），这时按文件名
    在本机的 voices/ 里找同名副本 —— 附加包的音色目录正好是这种情况。
    """
    if not p:
        return ""
    s = str(p)
    is_abs = os.path.isabs(s) or (len(s) > 1 and s[1] == ":")
    if is_abs and os.path.exists(s):
        return s
    cand = Path(s) if is_abs else ROOT / s
    if not is_abs and cand.exists():
        return str(cand)
    name = Path(s).name
    if name:
        for sub in ("bsai", "voices", "ref_audios"):
            base = ROOT / sub
            if not base.exists():
                continue
            for hit in base.rglob(name):
                if hit.is_file():
                    return str(hit)
    return s


def _normalize_voice(v: dict) -> dict:
    """统一音色条目：路径一律相对化，字段名对齐 bsai 代理层的 /api/voices。"""
    return {
        "name": v.get("name") or Path(str(v.get("ref_audio_path", "voice"))).stem,
        "description": v.get("description", ""),
        "text_lang": v.get("text_lang", "zh"),
        "prompt_lang": v.get("prompt_lang", "ja"),
        "ref_audio_path": _rel(v.get("ref_audio_path", "")),
        "prompt_text": v.get("prompt_text", ""),
    }


def _scan_voices() -> list:
    """没有 BSAI_VOICES.json 时，直接扫目录；同名 .txt 当 prompt_text。"""
    out, seen = [], set()
    for d in VOICE_SEARCH_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
                continue
            if p in seen:
                continue
            seen.add(p)
            prompt = ""
            side = p.with_suffix(".txt")
            if side.exists():
                try:
                    prompt = side.read_text(encoding="utf-8").strip()
                except Exception:
                    prompt = ""
            out.append({
                "name": _rel(p).rsplit(".", 1)[0].replace("/", "_"),
                "description": p.parent.name if p.parent != base else "",
                "text_lang": "zh",
                "prompt_lang": "ja" if prompt else "zh",
                "ref_audio_path": _rel(p),
                "prompt_text": prompt,
            })
    return out


# ---------------------------------------------------------------- 对外数据
def voices() -> list:
    data = _read_json(ROOT / VOICES_JSON)
    items = data.get("voices") if isinstance(data.get("voices"), list) else None
    if not items:
        items = _scan_voices()
    out = []
    for v in items:
        if not isinstance(v, dict):
            continue
        nv = _normalize_voice(v)
        if nv["ref_audio_path"]:
            out.append(nv)
    return out


def models() -> list:
    """补丁声明的模型 ∩ 后端真实存在的版本段。清单缺失时以真实版本为准。"""
    patch = _read_json(ROOT / PATCH_JSON)
    declared = [str(m) for m in (patch.get("models") or []) if isinstance(m, str)]
    present = _read_versions()
    if declared and present:
        keep = [m for m in declared if m in present]
    else:
        keep = declared or present
    return [{"name": m, "label": MODEL_LABELS.get(m, m),
             "sample_steps": 4 if m == "v5turbo" else None} for m in keep]


def capabilities() -> dict:
    patch = _read_json(ROOT / PATCH_JSON)
    vs = voices()
    ms = models()
    return {
        "server_name": os.environ.get("BSAI_SERVER_NAME") or patch.get("server_name") or ROOT.name,
        "backend": str(ROOT),
        "patch": patch.get("patch", ""),
        "features": patch.get("features") or {},
        # 前端认这个字段名：真实可用的模型版本名列表
        "backend_capabilities": [m["name"] for m in ms],
        "models": ms,
        "voices": vs,
        "voice_count": len(vs),
    }


def voices_payload() -> dict:
    """与 bsai 代理层 /api/voices 完全同构，前端可以无缝读任意一台。"""
    caps = capabilities()
    versions = caps["models"]
    return {
        "voices": [{"name": v["name"], "description": v["description"]} for v in caps["voices"]],
        "versions": versions,
        "default_version": versions[-1]["name"] if versions else None,
        "backend_capabilities": caps["backend_capabilities"],
        "server_name": caps["server_name"],
        "voice_details": caps["voices"],     # 带 ref_audio_path / prompt_text，便于直接引用
    }


def resolve_voice(ref) -> dict | None:
    """把「音色名 / 相对路径 / 绝对路径」统一解析成一轮请求要的参数。

    池服务与 api_v2 都用它：客户端只传 `voice: "v4_default"` 即可，
    不必知道参考音在本机的什么位置。
    """
    if not ref:
        return None
    target = str(ref)
    for v in voices():
        if v["name"] == target:
            return dict(v, ref_audio_path=_abs(v["ref_audio_path"]))
    for v in voices():                        # 也接受「只给文件名」的写法
        if Path(v["ref_audio_path"]).name == Path(target).name:
            return dict(v, ref_audio_path=_abs(v["ref_audio_path"]))
    p = _abs(target)
    if os.path.exists(p):
        return {"name": Path(p).stem, "ref_audio_path": p, "prompt_text": "",
                "prompt_lang": "ja", "text_lang": "zh"}
    return None


# ---------------------------------------------------------------- 挂载路由
def attach(app, prefix: str = "/api") -> None:
    """把三条只读路由挂到任意 FastAPI 应用上（api_v2 / 池服务都适用）。"""
    p = prefix.rstrip("/")

    @app.get(p + "/capabilities")
    def _bsai_capabilities():            # noqa: ANN202
        return capabilities()

    @app.get(p + "/models")
    def _bsai_models():                  # noqa: ANN202
        caps = capabilities()
        return {"models": caps["models"],
                "backend_capabilities": caps["backend_capabilities"],
                "server_name": caps["server_name"]}

    @app.get(p + "/voices")
    def _bsai_voices():                  # noqa: ANN202
        return voices_payload()


if __name__ == "__main__":               # 命令行自查：python bsai_capabilities.py
    print(json.dumps(capabilities(), ensure_ascii=False, indent=2))
