"""
本地 MiMo API mock —— 仅用于离线测试，不访问外网。

模拟小米 MiMo 开放平台的 chat/completions 变体 TTS 协议：
  POST /v1/chat/completions
    - 待合成文本放在 role=="assistant" 的消息里（没有就 400）
    - 风格指令在 role=="user" 消息里（记录日志，不影响输出）
    - audio.format: wav(默认) | mp3 | pcm | pcm16
    - 鉴权：请求头 api-key 或 Authorization: Bearer；"sk-bad" 一律 401
    - 响应：choices[0].message.audio.data = base64 音频
      （wav = 24kHz 16bit 单声道；pcm/pcm16 = 裸 PCM 字节）

错误路径（供测试断言）：
  - api-key 缺失或 "sk-bad"            → 401
  - model 不以 "mimo-" 开头            → 404
  - 无 assistant 消息                  → 400
  - 文本含 "FAIL" 或 "触发失败"        → 500

用法：
  python mock_mimo_server.py --port 9930     # 起服务（阻塞）
  python mock_mimo_server.py --selftest      # 起服务 + 自检后退出
"""

import argparse
import base64
import io
import logging
import threading
import time

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] mock-mimo: %(message)s")
log = logging.getLogger("mock-mimo")

app = FastAPI()
_REQ_N = [0]
_RECENT = []      # 最近 20 次合成请求的观测记录（测试断言转发内容用）
PRESET_MODELS = ["mimo-v2.5-tts", "mimo-v2.5-tts-voicedesign", "mimo-v2.5-tts-voiceclone"]


def _make_audio(fmt: str) -> bytes:
    """0.7s / 24kHz / 单声道：440Hz 正弦 + 首尾 50ms 余弦淡入淡出，幅值 0.4。

    幅值故意取大（RMS≈0.28），远超 server_intel 的静音阈值
    _SIL_SPEAK_RMS=0.010，保证合成结果不会被当成静音拒绝。
    """
    sr = 24000
    dur = 0.7
    t = np.arange(int(sr * dur), dtype=np.float64) / sr
    wave = 0.4 * np.sin(2 * np.pi * 440.0 * t)
    fade = int(sr * 0.05)
    env = np.ones_like(wave)
    if fade:
        env[:fade] = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
        env[-fade:] = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade, 0, -1) / fade)
    pcm = (wave * env * 32767.0).astype(np.int16)
    if fmt in ("pcm", "pcm16"):
        return pcm.tobytes()
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@app.get("/v1/models")
def models():
    """探活端点：server_intel 的 _engine_preflight 会 GET {url}/models。"""
    return {"object": "list",
            "data": [{"id": m, "object": "model"} for m in PRESET_MODELS]}


@app.get("/v1/_recent")
def recent():
    """测试观测端点（仅 mock 提供）：最近 20 次合成请求收到的风格指令等。"""
    return {"requests": _RECENT[-20:]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    api_key = request.headers.get("api-key") or ""
    if not api_key:
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            api_key = auth[7:].strip()
    if not api_key or api_key == "sk-bad":
        return JSONResponse(status_code=401,
                            content={"error": {"message": "Invalid API key"}})
    body = await request.json()
    model = str(body.get("model") or "")
    if not model.startswith("mimo-"):
        return JSONResponse(status_code=404,
                            content={"error": {"message": "model not found"}})
    text = ""
    style_text = ""
    for m in (body.get("messages") or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            text = str(m.get("content") or "")
        elif isinstance(m, dict) and m.get("role") == "user":
            style_text = str(m.get("content") or "")
            log.info("style instruction: %s", style_text)
    if not text:
        return JSONResponse(status_code=400,
                            content={"error": {"message": "assistant message required"}})
    if "FAIL" in text or "触发失败" in text:
        return JSONResponse(status_code=500,
                            content={"error": {"message": "mock internal error"}})
    audio = body.get("audio") or {}
    fmt = str(audio.get("format") or "wav").lower()
    if fmt not in ("wav", "mp3", "pcm", "pcm16"):
        fmt = "wav"
    _REQ_N[0] += 1
    _RECENT.append({"n": _REQ_N[0], "style": style_text,
                    "voice": audio.get("voice"), "format": fmt, "text": text})
    del _RECENT[:-20]
    data = base64.b64encode(_make_audio(fmt)).decode("ascii")
    log.info("tts #%d voice=%r fmt=%s text=%r", _REQ_N[0],
             audio.get("voice"), fmt, text[:30])
    return {
        "id": f"chatcmpl-mock-{_REQ_N[0]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": style_text,
                        "audio": {"id": f"audio-mock-{_REQ_N[0]}", "data": data,
                                  "expires_at": None, "transcript": None}},
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _serve(port: int):
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def _selftest(port: int) -> int:
    """起服务 → 打一轮请求自检（成功 + 4 种错误路径）→ 退出。"""
    import requests as _rq
    th = threading.Thread(target=_serve, args=(port,), daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(75):
        try:
            if _rq.get(base + "/v1/models", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.2)
    else:
        print("MOCK SELFTEST FAIL: server not ready")
        return 1
    try:
        r = _rq.post(base + "/v1/chat/completions", headers={"api-key": "sk-test"},
                     json={"model": "mimo-v2.5-tts",
                           "messages": [{"role": "user", "content": "开心一点"},
                                        {"role": "assistant", "content": "你好"}],
                           "audio": {"format": "wav", "voice": "冰糖"}})
        assert r.status_code == 200, f"synth -> {r.status_code}"
        wav = base64.b64decode(r.json()["choices"][0]["message"]["audio"]["data"])
        info = sf.info(io.BytesIO(wav))
        assert info.samplerate == 24000 and info.frames > 0, "bad wav"
        assert _rq.post(base + "/v1/chat/completions", headers={"api-key": "sk-bad"},
                        json={"model": "mimo-v2.5-tts",
                              "messages": [{"role": "assistant", "content": "x"}]}
                        ).status_code == 401, "bad key -> 401"
        assert _rq.post(base + "/v1/chat/completions", headers={"api-key": "sk-test"},
                        json={"model": "gpt-4",
                              "messages": [{"role": "assistant", "content": "x"}]}
                        ).status_code == 404, "bad model -> 404"
        assert _rq.post(base + "/v1/chat/completions", headers={"api-key": "sk-test"},
                        json={"model": "mimo-v2.5-tts",
                              "messages": [{"role": "user", "content": "x"}]}
                        ).status_code == 400, "no assistant -> 400"
        assert _rq.post(base + "/v1/chat/completions", headers={"api-key": "sk-test"},
                        json={"model": "mimo-v2.5-tts",
                              "messages": [{"role": "assistant", "content": "请FAIL"}]}
                        ).status_code == 500, "FAIL text -> 500"
    except Exception as e:
        print(f"MOCK SELFTEST FAIL: {e!r}")
        return 1
    print("MOCK SELFTEST PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="本地 MiMo API mock（仅供离线测试）")
    ap.add_argument("--port", type=int, default=9930)
    ap.add_argument("--selftest", action="store_true", help="自检后退出")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(_selftest(args.port))
    _serve(args.port)
