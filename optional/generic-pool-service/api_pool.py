"""GPT-SoVITS 多路并发 TTS 服务（进程池 + 批量/流式 + 后端中立 + IPv6 公网）

端点
  POST /                 单句合成 → audio/wav（字段兼容 api.py；服务器可预置默认参考音）
  POST /tts_batch        texts[] → {"sample_rate", "wavs": [b64...]}（顺序与 texts 一致）
  POST /tts_batch_stream texts[] → NDJSON，每完成一句回一行 {"index":k,"wav":"<b64>"}
  GET  /health           池状态与会话统计（兼容旧版）
  GET  /healthz          存活探针（永远 200）
  GET  /readyz           就绪探针（池未就绪时 503）
  GET  /docs             OpenAPI 文档

配置优先级：命令行 > 环境变量 > 配置文件 > 内置默认值
  配置文件默认读 config/pool_api.json（可用 --config-file 指定；也可用 config/pool_api.example.json 起步）

批量/流式说明：本实现用「多进程池」，每个 worker 自己拿下一句，
等价于 BSAIReader 补丁里"算好一句接下一句"的动态接活，只是把空槽补位放在进程层。

公网暴露：默认绑定 IPv6 双栈；启动时打印全局 IPv6、防火墙放行命令与鉴权状态。
"""

import argparse
import base64
import io
import json
import os
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from tts_pool import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    PoolBusy,
    TTSPool,
    max_pool_size,
    pick_gpu_index,
)

FIELD_MAP = {
    "refer_wav_path": "ref_audio_path",
    "prompt_language": "prompt_lang",
    "text_language": "text_lang",
    "speed": "speed_factor",
}

DEFAULTS = {
    "host": "::",
    "port": 9881,
    "pool_size": 4,
    "gpu": None,
    "device": "auto",
    "config": DEFAULT_CONFIG,
    "python": None,
    "api_key": "",
    "cors_origins": [],
    "max_text_chars": 0,
    "queue_timeout": 120,
    "request_timeout": 900,
    "warm": True,
    "warm_ref": None,
    "warm_text": "预热一下",
    "default_ref_audio": None,
    "default_prompt_text": None,
    "default_prompt_lang": "zh",
    "default_text_lang": "zh",
    "default_sample_steps": None,
    "per_instance_gb": 2.2,
    "startup_timeout": 420,
    "open_firewall": False,
}

ENV_MAP = {
    "POOL_HOST": ("host", str),
    "POOL_PORT": ("port", int),
    "POOL_SIZE": ("pool_size", int),
    "POOL_GPU": ("gpu", int),
    "POOL_DEVICE": ("device", str),
    "POOL_CONFIG": ("config", str),
    "POOL_PYTHON": ("python", str),
    "TTS_API_KEY": ("api_key", str),
    "POOL_MAX_TEXT": ("max_text_chars", int),
    "POOL_QUEUE_TIMEOUT": ("queue_timeout", float),
    "POOL_REQUEST_TIMEOUT": ("request_timeout", float),
    "POOL_DEFAULT_REF": ("default_ref_audio", str),
    "POOL_DEFAULT_PROMPT": ("default_prompt_text", str),
    "POOL_CORS": ("cors_origins", lambda v: [s.strip() for s in v.split(",") if s.strip()]),
}

STATS = {"served": 0, "inflight": 0, "max_inflight": 0, "batch_requests": 0, "sentences": 0,
         "rejected": 0, "failed": 0}
STATS_LOCK = threading.Lock()
POOL: TTSPool = None
CFG = dict(DEFAULTS)
API_KEY = ""


# ---------------------------------------------------------------- 配置
def load_settings():
    """合并 默认值 < 配置文件 < 环境变量，返回 dict。"""
    cfg = dict(DEFAULTS)
    cfg_path = os.environ.get("POOL_CONFIG_FILE") or os.path.join(PROJECT_ROOT, "config", "pool_api.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                data = json.load(f)
            unknown = set(data) - set(DEFAULTS)
            if unknown:
                print(f"[api_pool] 配置文件里有未知字段（已忽略）: {sorted(unknown)}", flush=True)
            cfg.update({k: v for k, v in data.items() if k in DEFAULTS})
            cfg["_loaded_from"] = cfg_path
        except Exception as exc:
            print(f"[api_pool] 配置文件读取失败 {cfg_path}: {exc!r}", flush=True)
    for env, (key, cast) in ENV_MAP.items():
        raw = os.environ.get(env)
        if raw is None or raw == "":
            continue
        try:
            cfg[key] = cast(raw)
        except Exception:
            print(f"[api_pool] 环境变量 {env}={raw!r} 解析失败，已忽略", flush=True)
    return cfg


def apply_cli(cfg, argv=None):
    p = argparse.ArgumentParser(description="GPT-SoVITS 多路并发 API（进程池 + 批量/流式 + IPv6）")
    p.add_argument("--config-file", default=None, help="配置文件路径（默认 config/pool_api.json）")
    p.add_argument("-p", "--port", type=int, default=None)
    p.add_argument("-s", "--pool-size", type=int, default=None, help="实例数（默认 4）")
    p.add_argument("-g", "--gpu", type=int, default=None, help="物理 GPU 序号；不传=自动（跳过核显）")
    p.add_argument("--device", default=None,
                   help="auto|cuda|cuda:N|xpu|mps|dml|cpu（默认 auto，交给 device_utils 解析）")
    p.add_argument("-c", "--config", dest="tts_config", default=None, help="tts_infer.yaml 路径")
    p.add_argument("--python", default=None, help="worker 解释器路径（默认当前解释器）")
    p.add_argument("--host", default=None, help="绑定地址，默认 :: 表示 IPv6 双栈（含 IPv4）")
    p.add_argument("--no-ipv6", action="store_true", help="只绑定 IPv4（0.0.0.0）")
    p.add_argument("--api-key", default=None,
                   help='鉴权 key；传 "auto" 自动生成；留空=不鉴权（公网暴露时强烈建议设置）')
    p.add_argument("--cors", default=None, help='允许的跨域来源，逗号分隔；"*" 表示全部')
    p.add_argument("--max-text-chars", type=int, default=None, help="单句最大字符数，0=不限制")
    p.add_argument("--queue-timeout", type=float, default=None, help="池满时等待空闲实例的秒数，超时返回 503")
    p.add_argument("--request-timeout", type=float, default=None, help="单次合成最长等待秒数")
    p.add_argument("--open-firewall", action="store_true", help="尝试用 netsh 放行端口（需管理员）")
    p.add_argument("--warm-ref", default=None, help="启动预热用的参考音（默认自动找）")
    p.add_argument("--no-warm", action="store_true", help="跳过预热（首次请求会慢约 18s/实例）")
    p.add_argument("--default-ref", default=None, help="服务端默认参考音，客户端可不传 refer_wav_path")
    p.add_argument("--default-prompt", default=None, help="默认参考音对应的文本")
    p.add_argument("--default-lang", default=None, help="默认参考/文本语种（zh/en/ja/...）")
    p.add_argument("--default-steps", type=int, default=None, help="默认采样步数，客户端可不传")
    p.add_argument("--per-instance-gb", type=float, default=None, help="每个实例的显存估算（用于池大小上限）")
    args = p.parse_args(argv)

    if args.config_file:
        os.environ["POOL_CONFIG_FILE"] = args.config_file
        cfg.clear()
        cfg.update(load_settings())
    if args.port is not None:
        cfg["port"] = args.port
    if args.pool_size is not None:
        cfg["pool_size"] = args.pool_size
    if args.gpu is not None:
        cfg["gpu"] = args.gpu
    if args.device is not None:
        cfg["device"] = args.device
    if args.tts_config is not None:
        cfg["config"] = args.tts_config
    if args.python is not None:
        cfg["python"] = args.python
    if args.host is not None:
        cfg["host"] = args.host
    if args.no_ipv6:
        cfg["host"] = "0.0.0.0"
    if args.api_key is not None:
        cfg["api_key"] = args.api_key
    if args.cors is not None:
        cfg["cors_origins"] = [s.strip() for s in args.cors.split(",") if s.strip()]
    if args.max_text_chars is not None:
        cfg["max_text_chars"] = args.max_text_chars
    if args.queue_timeout is not None:
        cfg["queue_timeout"] = args.queue_timeout
    if args.request_timeout is not None:
        cfg["request_timeout"] = args.request_timeout
    if args.per_instance_gb is not None:
        cfg["per_instance_gb"] = args.per_instance_gb
    if args.open_firewall:
        cfg["open_firewall"] = True
    if args.warm_ref is not None:
        cfg["warm_ref"] = args.warm_ref
    if args.no_warm:
        cfg["warm"] = False
    if args.default_ref is not None:
        cfg["default_ref_audio"] = args.default_ref
    if args.default_prompt is not None:
        cfg["default_prompt_text"] = args.default_prompt
    if args.default_lang is not None:
        cfg["default_prompt_lang"] = args.default_lang
        cfg["default_text_lang"] = args.default_lang
    if args.default_steps is not None:
        cfg["default_sample_steps"] = args.default_steps
    return cfg


# ---------------------------------------------------------------- 工具
def _base_inputs(body: dict) -> dict:
    """请求体 → TTS.run 输入（不含 text 的校验）；缺省字段用服务器配置补齐。"""
    out = {}
    for k, v in body.items():
        if k in ("texts", "api_key") or v is None:
            continue
        out[FIELD_MAP.get(k, k)] = v
    if "ref_audio_path" not in out and CFG.get("default_ref_audio"):
        out["ref_audio_path"] = CFG["default_ref_audio"]
        if CFG.get("default_prompt_text"):
            out.setdefault("prompt_text", CFG["default_prompt_text"])
        out.setdefault("prompt_lang", CFG.get("default_prompt_lang") or "zh")
    out.setdefault("text_lang", CFG.get("default_text_lang") or "zh")
    if out.get("sample_steps") is None and CFG.get("default_sample_steps"):
        out["sample_steps"] = int(CFG["default_sample_steps"])
    if "ref_audio_path" not in out:
        raise ValueError("缺少 refer_wav_path（服务器未配置 default_ref_audio）")
    return out


def _to_inputs(body: dict) -> dict:
    """单句：在 _base_inputs 之上要求 text 并检查长度。"""
    out = _base_inputs(body)
    text = out.get("text")
    if not text:
        raise ValueError("缺少 text")
    limit = int(CFG.get("max_text_chars") or 0)
    if limit > 0 and len(str(text)) > limit:
        raise ValueError(f"text 超过上限 {limit} 字符")
    return out


def _batch_jobs(texts, body) -> list:
    base = _base_inputs({k: v for k, v in body.items() if k != "texts"})
    base.pop("text", None)
    jobs = []
    limit = int(CFG.get("max_text_chars") or 0)
    for text in texts:
        if not text:
            raise ValueError("texts 里有空字符串")
        if limit > 0 and len(str(text)) > limit:
            raise ValueError(f"texts 中含超过 {limit} 字符的句子")
        job = dict(base)
        job["text"] = text
        jobs.append(job)
    return jobs


def _wav_bytes(sr: int, audio: np.ndarray) -> bytes:
    arr = np.asarray(audio)
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    buf = io.BytesIO()
    sf.write(buf, arr, int(sr), format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _b64_wav(sr: int, audio: np.ndarray) -> str:
    return base64.b64encode(_wav_bytes(sr, audio)).decode("ascii")


def _auth_error(request: Request):
    if not API_KEY:
        return None
    provided = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if provided == API_KEY:
        return None
    return JSONResponse({"error": "unauthorized: X-API-Key required"}, status_code=401)


def _stat(served=0, sentences=0, batch=0, rejected=0, failed=0, delta_inflight=0):
    with STATS_LOCK:
        STATS["served"] += served
        STATS["sentences"] += sentences
        STATS["batch_requests"] += batch
        STATS["rejected"] += rejected
        STATS["failed"] += failed
        STATS["inflight"] += delta_inflight
        STATS["max_inflight"] = max(STATS["max_inflight"], STATS["inflight"])


def _rid() -> str:
    return uuid.uuid4().hex[:12]


def _error(status: int, message: str, rid: str, extra=None, retry_after=None):
    payload = {"error": message, "request_id": rid}
    if extra:
        payload.update(extra)
    headers = {"X-Request-Id": rid}
    if retry_after:
        headers["Retry-After"] = str(retry_after)
    return JSONResponse(payload, status_code=status, headers=headers)


def _global_ipv6():
    found = []
    try:
        import psutil

        for ifname, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET6:
                    ip = addr.address.split("%")[0]
                    low = ip.lower()
                    if low.startswith("fe80") or low in ("::1", "::"):
                        continue
                    found.append((ifname, ip))
    except Exception:
        pass
    if not found:
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET6):
                ip = info[4][0].split("%")[0]
                low = ip.lower()
                if low.startswith("fe80") or low in ("::1", "::"):
                    continue
                if ip not in [x[1] for x in found]:
                    found.append(("host", ip))
        except Exception:
            pass
    return found


def _make_socket(host: str, port: int):
    if ":" in host:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError:
            pass
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(2048)
    return sock


def _open_firewall(port: int):
    import subprocess

    cmd = (
        f'netsh advfirewall firewall add rule name="GPT-SoVITS Pool API {port}" '
        f"dir=in action=allow protocol=TCP localport={port}"
    )
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        print(f"[api_pool] 防火墙放行{'成功' if out.returncode == 0 else '失败（可能不是管理员）'}", flush=True)
    except Exception as exc:
        print(f"[api_pool] 防火墙放行出错: {exc!r}", flush=True)


def _default_warm_ref():
    """自动找一段现成参考音做预热（可用 --warm-ref / warm_ref 覆盖）。"""
    import glob

    if CFG.get("default_ref_audio") and os.path.exists(CFG["default_ref_audio"]):
        return CFG["default_ref_audio"]
    for pattern in ("ref_audios/*.wav", "SunFLV/Reference_Voices/*.wav", "SunFLV/Baked_Voices/*.wav",
                    "output/**/*.wav"):
        hits = sorted(glob.glob(os.path.join(PROJECT_ROOT, pattern), recursive=True))
        if hits:
            return hits[0]
    return None


# ---------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        if POOL is not None:
            POOL.close()


app = FastAPI(title="GPT-SoVITS Pool API", version="2.0", lifespan=lifespan)


class DynamicCORS:
    """按请求读取 CFG 的 CORS（因为配置可能在 main() 里才由 CLI 补齐）。"""

    def __init__(self, app):
        self.app = app

    def _origins(self):
        return CFG.get("cors_origins") or []

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        origins = self._origins()
        if not origins:
            return await self.app(scope, receive, send)

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        origin = headers.get("origin", "")
        allow_all = "*" in origins
        allowed = allow_all or origin in origins
        if not allowed:
            return await self.app(scope, receive, send)

        acao = (b"*", b"*") if allow_all else (b"access-control-allow-origin", origin.encode())
        extra = [
            acao,
            (b"access-control-expose-headers",
             b"X-Request-Id, X-Elapsed, X-Audio-Seconds, X-Sample-Rate"),
        ]
        if scope["method"] == "OPTIONS":
            base = [(b"access-control-allow-origin", b"*" if allow_all else origin.encode()),
                    (b"access-control-allow-methods", b"*"),
                    (b"access-control-allow-headers", b"*"),
                    (b"access-control-max-age", b"600")]
            await send({"type": "http.response.start", "status": 204, "headers": base})
            await send({"type": "http.response.body", "body": b""})
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + extra
            await send(message)

        return await self.app(scope, receive, send_wrapper)


app.add_middleware(DynamicCORS)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    ok = POOL is not None and POOL.alive == POOL.size
    return JSONResponse({"status": "ready" if ok else "starting"}, status_code=200 if ok else 503)


@app.get("/health")
def health():
    with STATS_LOCK:
        return {
            "pool_size": POOL.size if POOL else 0,
            "alive": POOL.alive if POOL else 0,
            "devices": POOL.devices if POOL else [],
            "startup_seconds": round(POOL.startup_seconds, 1) if POOL else None,
            "served": STATS["served"],
            "sentences": STATS["sentences"],
            "batch_requests": STATS["batch_requests"],
            "rejected": STATS["rejected"],
            "failed": STATS["failed"],
            "inflight": STATS["inflight"],
            "max_inflight": STATS["max_inflight"],
            "auth": "api-key" if API_KEY else "open",
            "defaults": {
                "ref_audio": bool(CFG.get("default_ref_audio")),
                "text_lang": CFG.get("default_text_lang"),
                "sample_steps": CFG.get("default_sample_steps"),
                "max_text_chars": CFG.get("max_text_chars"),
            },
        }


@app.get("/")
def index():
    return {
        "service": "GPT-SoVITS Pool API",
        "version": "2.0",
        "endpoints": {
            "POST /": "单句合成 → audio/wav",
            "POST /tts_batch": "texts[] → {sample_rate, wavs[b64]}",
            "POST /tts_batch_stream": 'texts[] → NDJSON {"index","wav"}',
            "GET /health": "池状态与统计",
            "GET /healthz": "存活探针",
            "GET /readyz": "就绪探针",
            "GET /docs": "OpenAPI",
        },
        "auth": "api-key" if API_KEY else "open",
        "server_defaults": {
            "ref_audio": CFG.get("default_ref_audio"),
            "prompt_text": CFG.get("default_prompt_text"),
            "text_lang": CFG.get("default_text_lang"),
            "sample_steps": CFG.get("default_sample_steps"),
        },
    }


@app.post("/")
async def tts(request: Request):
    rid = _rid()
    denied = _auth_error(request)
    if denied is not None:
        _stat(rejected=1)
        return denied
    body = await request.json()
    try:
        inputs = _to_inputs(body)
    except ValueError as exc:
        _stat(rejected=1)
        return _error(400, str(exc), rid)

    _stat(delta_inflight=1)

    def _work():
        t0 = time.perf_counter()
        sr, audio = POOL.synthesize(wait_timeout=CFG.get("queue_timeout"), **inputs)
        return sr, audio, time.perf_counter() - t0

    try:
        sr, audio, elapsed = await run_in_threadpool(_work)
    except PoolBusy as exc:
        _stat(rejected=1)
        return _error(503, str(exc), rid, retry_after=5)
    except Exception as exc:
        _stat(failed=1)
        return _error(500, f"{type(exc).__name__}: {exc}", rid)
    finally:
        _stat(served=1, delta_inflight=-1)

    return Response(
        content=_wav_bytes(sr, audio),
        media_type="audio/wav",
        headers={
            "X-Request-Id": rid,
            "X-Sample-Rate": str(sr),
            "X-Elapsed": f"{elapsed:.3f}",
            "X-Audio-Seconds": f"{len(np.asarray(audio)) / float(sr):.3f}",
        },
    )


@app.post("/tts_batch")
async def tts_batch(request: Request):
    rid = _rid()
    denied = _auth_error(request)
    if denied is not None:
        _stat(rejected=1)
        return denied
    body = await request.json()
    texts = body.get("texts") or []
    if not texts:
        _stat(rejected=1)
        return _error(400, "texts is required", rid)
    try:
        jobs = _batch_jobs(texts, body)
    except ValueError as exc:
        _stat(rejected=1)
        return _error(400, str(exc), rid)

    _stat(batch=1)

    def _work():
        return POOL.synthesize_many(jobs, wait_timeout=CFG.get("queue_timeout"))

    results = await run_in_threadpool(_work)

    wavs, errors, sample_rate = [], [], None
    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            errors.append({"index": idx, "error": str(res)})
            wavs.append("")
            continue
        sr, audio = res
        sample_rate = sr if sample_rate is None else sample_rate
        wavs.append(_b64_wav(sr, audio))
    _stat(served=len(texts) - len(errors), sentences=len(texts), failed=len(errors))

    if len(errors) == len(texts):
        return _error(500, "all sentences failed", rid, extra={"errors": errors})
    payload = {"sample_rate": sample_rate, "wavs": wavs}
    if errors:
        payload["errors"] = errors
    return JSONResponse(payload, headers={"X-Request-Id": rid})


@app.post("/tts_batch_stream")
async def tts_batch_stream(request: Request):
    rid = _rid()
    denied = _auth_error(request)
    if denied is not None:
        _stat(rejected=1)
        return denied
    body = await request.json()
    texts = body.get("texts") or []
    if not texts:
        _stat(rejected=1)
        return _error(400, "texts is required", rid)
    try:
        jobs = _batch_jobs(texts, body)
    except ValueError as exc:
        _stat(rejected=1)
        return _error(400, str(exc), rid)

    _stat(batch=1)

    def gen():
        with ThreadPoolExecutor(max_workers=POOL.size) as pool_exec:
            futures = {
                pool_exec.submit(POOL.synthesize, wait_timeout=CFG.get("queue_timeout"), **job): i
                for i, job in enumerate(jobs)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    sr, audio = fut.result()
                    _stat(served=1)
                    yield json.dumps(
                        {"index": idx, "sample_rate": sr, "wav": _b64_wav(sr, audio)},
                        ensure_ascii=True,
                    ) + "\n"
                except Exception as exc:
                    _stat(failed=1)
                    yield json.dumps(
                        {"index": idx, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=True
                    ) + "\n"
            _stat(sentences=len(texts))

    return StreamingResponse(
        gen(), media_type="application/x-ndjson", headers={"X-Request-Id": rid}
    )


# ---------------------------------------------------------------- main
def main(argv=None):
    global CFG, POOL, API_KEY

    # 先读环境/文件，再让 CLI 覆盖；--config-file 由 CLI 解析时生效
    pre = dict(load_settings())
    CFG = pre
    CFG = apply_cli(CFG, argv)
    CFG.setdefault("_loaded_from", None)

    API_KEY = str(CFG.get("api_key") or "")
    if API_KEY == "auto":
        API_KEY = base64.urlsafe_b64encode(os.urandom(18)).decode().rstrip("=")
    host = CFG["host"]
    port = int(CFG["port"])

    gpu = CFG.get("gpu")
    if gpu is None:
        gpu = pick_gpu_index()
    cap = max_pool_size(gpu, per_instance_gb=float(CFG.get("per_instance_gb") or 2.2))
    size = max(1, min(int(CFG["pool_size"]), cap)) if gpu is not None else int(CFG["pool_size"])
    if size != int(CFG["pool_size"]):
        print(f"[api_pool] pool_size {CFG['pool_size']} 超出显存允许的 {cap}，已降为 {size}", flush=True)

    warm_ref = None
    if CFG.get("warm"):
        warm_ref = CFG.get("warm_ref") or _default_warm_ref()

    t0 = time.time()
    POOL = TTSPool(
        size=size,
        config_path=CFG["config"],
        gpu=gpu,
        device=CFG.get("device") or "auto",
        python=CFG.get("python"),
        warm_ref=warm_ref,
        warm_text=CFG.get("warm_text") or "预热一下",
        warm_prompt=CFG.get("default_prompt_text"),
        per_instance_gb=float(CFG.get("per_instance_gb") or 2.2),
        spawn_timeout=float(CFG.get("startup_timeout") or 420),
        request_timeout=float(CFG.get("request_timeout") or 900),
    )

    sock = _make_socket(host, port)
    v6 = ":" in host
    print("=" * 64, flush=True)
    print(f"[api_pool] 就绪：{size} 个实例 / device={CFG.get('device')} / gpu={gpu} / "
          f"启动 {time.time() - t0:.1f}s", flush=True)
    if CFG.get("_loaded_from"):
        print(f"[api_pool] 配置来自: {CFG['_loaded_from']}（命令行为准）", flush=True)
    if v6:
        print(f"[api_pool] 监听 [::]:{port}（IPv6 双栈，同时接受 IPv4）", flush=True)
        print(f"[api_pool]   本机   http://[::1]:{port}/readyz", flush=True)
        for ifname, ip in _global_ipv6()[:4]:
            print(f"[api_pool]   公网   http://[{ip}]:{port}/readyz   ({ifname})", flush=True)
    else:
        print(f"[api_pool] 监听 {host}:{port}（仅 IPv4）  http://127.0.0.1:{port}/readyz", flush=True)
    print(f"[api_pool] 鉴权: {'需要 X-API-Key' if API_KEY else '未启用（公网暴露请加 --api-key auto）'}", flush=True)
    if API_KEY:
        print(f"[api_pool] API Key: {API_KEY}", flush=True)
    if CFG.get("default_ref_audio"):
        print(f"[api_pool] 默认参考音: {CFG['default_ref_audio']}"
              f"（客户端可只传 text）", flush=True)
    if CFG.get("cors_origins"):
        print(f"[api_pool] CORS: {CFG['cors_origins']}", flush=True)
    print(f"[api_pool] 限流: queue_timeout={CFG.get('queue_timeout')}s "
          f"max_text_chars={CFG.get('max_text_chars')}", flush=True)
    print(f"[api_pool] 防火墙放行（管理员 CMD）:", flush=True)
    print(f'    netsh advfirewall firewall add rule name="GPT-SoVITS Pool API {port}" '
          f"dir=in action=allow protocol=TCP localport={port}", flush=True)
    print("=" * 64, flush=True)
    if CFG.get("open_firewall"):
        _open_firewall(port)

    import uvicorn

    config = uvicorn.Config(app, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    try:
        server.run(sockets=[sock])
    finally:
        try:
            sock.close()
        except Exception:
            pass
        POOL.close()
        print("[api_pool] 已关闭实例池", flush=True)


if __name__ == "__main__":
    main()
