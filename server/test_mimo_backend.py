"""
MiMo 云端引擎离线测试套件 —— 全部打本地 mock（127.0.0.1:9930），不访问外网。

用法（项目 runtime 解释器，依赖都在里面）：
  D:/BSAIReader/GPT-SoVITS-For-Intel/runtime/python.exe server/test_mimo_backend.py

覆盖：
  1. _engine_config：type/model/audio_format/rpm_limit/emotion_instructions 解析
  2. api_key 的 ${ENV} 展开与 mimo 的 MIMO_API_KEY 回落
  3. _engine_headers：mimo 用 api-key 头，gptsovits 用 X-API-Key
  4. _mimo_style_instruction：语气标签 / 语速翻译 / 未登记标签的通用兜底
  5. _mimo_synth / _mimo_synth_batch：成功、鉴权失败、错误路径
  6. _concat_wav：混采样率重采样 + 立体声压单声道
  7. 端到端（TestClient，不占新端口）：/api/servers、/api/tts、/api/tts/long 全链路
"""

import base64
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent.absolute()
MOCK_PORT = 9930
MOCK_BASE = f"http://127.0.0.1:{MOCK_PORT}"

# 必须在 import server_intel 之前设置：配置、任务缓存隔离、${ENV} 展开用的变量
os.environ["BSAI_CONFIG"] = str(HERE / "config_mimo_test.yaml")
os.environ["BSAI_TASK_CACHE"] = tempfile.mkdtemp(prefix="bsai_mimo_test_")
os.environ["MIMO_TEST_KEY"] = "sk-from-env"

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import requests  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# 嵌入式 runtime 不一定自动把脚本目录放进 sys.path，显式加一次
sys.path.insert(0, str(HERE))
import server_intel as si  # noqa: E402

_PASS, _FAIL = [], []


def check(name, fn):
    try:
        fn()
        _PASS.append(name)
        print(f"PASS  {name}")
    except Exception as e:
        _FAIL.append(name)
        print(f"FAIL  {name}: {e!r}")


def _make_wav(sr, dur, channels=1, freq=440.0):
    t = np.arange(int(sr * dur), dtype=np.float64) / sr
    data = 0.4 * np.sin(2 * np.pi * freq * t)
    if channels == 2:
        data = np.stack([data, data], axis=1)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _wav_sr(wav_bytes):
    return sf.info(io.BytesIO(wav_bytes)).samplerate


def _start_mock():
    proc = subprocess.Popen(
        [sys.executable, str(HERE / "mock_mimo_server.py"), "--port", str(MOCK_PORT)],
        cwd=str(HERE))
    for _ in range(75):
        try:
            if requests.get(MOCK_BASE + "/v1/models", timeout=1).status_code == 200:
                return proc
        except Exception:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("mock server not ready in time")


# ── 单元测试 ─────────────────────────────────────────────────────
ENG = si._engine_config("MiMo测试引擎")


def t_engine_config():
    assert ENG["type"] == "mimo", ENG
    assert ENG["api_key"] == "sk-test-mock", ENG["api_key"]
    assert ENG["model"] == "mimo-v2.5-tts"
    assert ENG["audio_format"] == "wav"
    assert ENG["rpm_limit"] == 600
    assert "开心" in ENG["emotion_instructions"]
    assert ENG["url"] == MOCK_BASE + "/v1"


def t_engine_config_env_expand():
    eng = si._engine_config("MiMo环境变量引擎")
    assert eng["api_key"] == "sk-from-env", eng["api_key"]


def t_engine_headers():
    h = si._engine_headers(ENG)
    assert h.get("api-key") == "sk-test-mock", h
    assert "X-API-Key" not in h, h
    h2 = si._engine_headers({"type": "gptsovits", "api_key": "x"})
    assert h2.get("X-API-Key") == "x" and "api-key" not in h2, h2


def t_style_instruction():
    assert "开心" in si._mimo_style_instruction(ENG, "开心", 1.0)
    assert "语速" in si._mimo_style_instruction(ENG, "", 1.5)
    assert si._mimo_style_instruction(ENG, "", 1.0) == ""
    # 语气 + 语速叠加成一条指令
    both = si._mimo_style_instruction(ENG, "火海", 1.5)
    assert "恨意" in both and "语速" in both, both
    # 未登记的语气标签 → 通用兜底指令
    assert "疑惑" in si._mimo_style_instruction(ENG, "疑惑", 1.0)
    # 空串指令（如 电话: ""）不产生 user 消息内容
    eng2 = dict(ENG, emotion_instructions={"电话": ""})
    assert si._mimo_style_instruction(eng2, "电话", 1.0) == ""


def t_mimo_synth_ok():
    wav = si._mimo_synth(ENG, "你好，世界。", "v4_default", 1.0)
    assert isinstance(wav, bytes) and len(wav) > 100
    assert _wav_sr(wav) == 24000, _wav_sr(wav)


def t_mimo_synth_fail_text():
    try:
        si._mimo_synth(ENG, "这句要FAIL", "v4_default", 1.0)
    except HTTPException as e:
        assert "500" in str(e.detail) or "服务端异常" in str(e.detail), e.detail
        return
    raise AssertionError("FAIL text should raise HTTPException")


def t_mimo_synth_bad_key():
    eng = dict(ENG, api_key="sk-bad")
    try:
        si._mimo_synth(eng, "你好", "v4_default", 1.0)
    except HTTPException as e:
        assert "401" in str(e.detail), e.detail
        return
    raise AssertionError("bad key should raise HTTPException")


def t_mimo_synth_missing_key():
    eng = dict(ENG, api_key="")
    try:
        si._mimo_synth(eng, "你好", "v4_default", 1.0)
    except HTTPException as e:
        assert "401" in str(e.detail), e.detail
        return
    raise AssertionError("missing key should raise HTTPException")


def t_mimo_synth_batch():
    out = si._mimo_synth_batch(ENG, ["第一句。", "第二句。"], "v4_default", 1.0)
    assert isinstance(out, list) and len(out) == 2, out
    assert all(isinstance(w, bytes) and w for w in out), out
    # 含失败句的批次：失败位置是 None，其余仍是 bytes
    out2 = si._mimo_synth_batch(ENG, ["正常句。", "这句要FAIL"], "v4_default", 1.0)
    assert isinstance(out2[0], bytes) and out2[1] is None, out2


def t_engine_synth_dispatch():
    # _engine_synth/_engine_synth_batch 顶部按 type 分流到 MiMo 适配器
    wav = si._engine_synth(ENG, "分流测试。", "v4_default", 1.0, pad_short=True)
    assert isinstance(wav, bytes) and _wav_sr(wav) == 24000
    wavs = si._engine_synth_batch(ENG, ["甲句。", "乙句。"], "v4_default", 1.0)
    assert len(wavs) == 2 and all(w for w in wavs), wavs


def t_concat_wav_resample():
    a = _make_wav(24000, 0.3)               # 第一块定基准 24k
    b = _make_wav(48000, 0.3)               # 48k 块要被重采样（旧逻辑会整块丢弃）
    c = _make_wav(24000, 0.2, channels=2)   # 立体声块压成单声道
    out, sr = si._concat_wav(a, b, c)
    assert sr == 24000, sr
    dur = si._wav_seconds(out.getvalue())
    assert abs(dur - 0.8) < 0.05, dur       # 0.3 + 0.3 + 0.2


# ── 端到端（TestClient 直连 app，不占新端口）────────────────────
CLIENT = TestClient(si.app)


def t_e2e_servers():
    r = CLIENT.get("/api/servers")
    assert r.status_code == 200, r.text
    servers = r.json()["servers"]
    mimos = [s for s in servers if s["engine"]]
    assert len(mimos) == 2, servers
    assert all(s["kind"] == "cloud" for s in mimos), mimos
    assert mimos[0]["name"] == "MiMo测试引擎"


def t_e2e_voices_local():
    # 勾选云端引擎时 /api/voices 仍由本机应答（voice_map 映射方案）
    r = CLIENT.get("/api/voices")
    assert r.status_code == 200, r.text
    names = [v["name"] for v in r.json()["voices"]]
    assert "v4_default" in names and "v4_phone" in names, names


def t_e2e_single_tts():
    r = CLIENT.post("/api/tts?_srv=0",
                    json={"text": "你好，世界。", "voice": "v4_default", "speed": 1.0})
    assert r.status_code == 200, r.text
    assert "wav" in r.headers.get("content-type", ""), r.headers.get("content-type")
    assert _wav_sr(r.content) == 24000
    # 带语气标签的请求也走通（emotion → user 风格指令）
    r2 = CLIENT.post("/api/tts?_srv=0",
                     json={"text": "今天天气不错。", "voice": "v4_default",
                           "speed": 1.0, "emotion": "开心"})
    assert r2.status_code == 200, r2.text


def t_e2e_long_task():
    # 注意：分句器会把太短的句子合并，这里用 4 个足够长的句子确保切成多句；
    # 断言只要求 ≥2 句，避免与分句启发式耦合。
    text = ("夜色像一块柔软的绒布盖住了整座城市，街灯次第亮起。"
            "高松灯抱紧了怀里的吉他，一步一步慢慢走上舞台。"
            "台下的灯牌一盏一盏亮了起来，像散落在人间的星星。"
            "她深吸了一口气，终于唱出了第一句歌词。")
    r = CLIENT.post("/api/tts/long?_srv=0",
                    json={"text": text, "voice": "v4_default", "speed": 1.0})
    assert r.status_code == 200, r.text
    tid = r.json()["task_id"]
    assert "MiMo" in r.json().get("engine", ""), r.json()
    deadline, p = time.time() + 90, {}
    while time.time() < deadline:
        pr = CLIENT.get(f"/api/tts/progress/{tid}")
        assert pr.status_code == 200, pr.text
        p = pr.json()
        if p.get("status") in ("done", "error"):
            break
        time.sleep(0.5)
    assert p.get("status") == "done", f"task end status={p}"
    res = CLIENT.get(f"/api/tts/result/{tid}")
    assert res.status_code == 200, res.text
    assert _wav_sr(res.content) == 24000
    sp = CLIENT.get(f"/api/tts/split/{tid}").json()
    total = sp.get("total") or 0
    assert total >= 2, total
    assert len(sp.get("starts") or []) == total, sp.get("starts")


def _last_mock_request():
    """mock 的观测端点：最近一次合成请求收到的内容（断言转发用）。"""
    r = requests.get(MOCK_BASE + "/v1/_recent", timeout=5)
    reqs = r.json().get("requests") or []
    assert reqs, "mock has no recorded requests"
    return reqs[-1]


def t_api_emotion_label_forwarded():
    # 请求体 emotion 标签 → MiMo 引擎 emotion_instructions → 风格指令
    r = CLIENT.post("/api/tts?_srv=0",
                    json={"text": "今天天气不错。", "voice": "v4_default",
                          "emotion": "开心"})
    assert r.status_code == 200, r.text
    last = _last_mock_request()
    assert "开心" in last["style"], last


def t_api_emotion_header_accepted():
    # 情感请求头：X-BSAI-Emotion（HTTP 头只能带 ASCII，非 ASCII 标签百分号编码）
    r = CLIENT.post("/api/tts?_srv=0", headers={"X-BSAI-Emotion": "%E7%81%AB%E6%B5%B7"},
                    json={"text": "风暴逼近。", "voice": "v4_default"})
    assert r.status_code == 200, r.text
    last = _last_mock_request()
    assert "恨意" in last["style"], last


def t_api_emotion_query_accepted():
    # 查询参数 ?emotion= 也能被接受（优先级低于请求体/请求头）
    r = CLIENT.post("/api/tts?_srv=0&emotion=%E5%BC%80%E5%BF%83",
                    json={"text": "出发吧。", "voice": "v4_default"})
    assert r.status_code == 200, r.text
    assert "开心" in _last_mock_request()["style"], _last_mock_request()


def t_api_style_field_forwarded():
    # 自由文本风格指令（style 字段）：MiMo 专属通道，与本地情绪参考音体系隔离
    r = CLIENT.post("/api/tts?_srv=0",
                    json={"text": "夜深了。", "voice": "v4_default",
                          "style": "用低沉沙哑的嗓音慢慢地读"})
    assert r.status_code == 200, r.text
    last = _last_mock_request()
    assert last["style"] == "用低沉沙哑的嗓音慢慢地读", last


def t_api_style_header_accepted():
    # 风格指令请求头：X-BSAI-Style（非 ASCII 百分号编码）
    from urllib.parse import quote
    r = CLIENT.post("/api/tts?_srv=0",
                    headers={"X-BSAI-Style": quote("像新闻主播一样播报")},
                    json={"text": "快讯一条。", "voice": "v4_default"})
    assert r.status_code == 200, r.text
    assert _last_mock_request()["style"] == "像新闻主播一样播报", _last_mock_request()


def t_api_style_overrides_emotion():
    # style 与 emotion 同时给：style 优先（emotion 标签不进指令）
    r = CLIENT.post("/api/tts?_srv=0",
                    json={"text": "走吧。", "voice": "v4_default",
                          "emotion": "开心", "style": "温柔一点，像哄孩子一样"})
    assert r.status_code == 200, r.text
    last = _last_mock_request()
    assert "温柔一点" in last["style"] and "开心" not in last["style"], last


def t_voices_engine_emotions_isolated():
    # /api/voices 用独立字段 engine_emotions 暴露 MiMo 自己的情绪词汇表，
    # 与本地音色的 emotions（参考音库）互不混用
    r = CLIENT.get("/api/voices?_srv=0")
    assert r.status_code == 200, r.text
    d = r.json()
    ee = d.get("engine_emotions") or {}
    assert "开心" in ee and "火海" in ee, ee
    assert isinstance(ee["开心"], str) and ee["开心"], ee
    # 本地音色的 emotions 字段保持原样（测试配置的 v4_default 无情绪库 → 空列表）
    v4 = next(v for v in d["voices"] if v["name"] == "v4_default")
    assert v4["emotions"] == [], v4


def t_style_long_task_forwarded():
    # 长文任务的 style 也要逐句转发到 MiMo
    r = CLIENT.post("/api/tts/long?_srv=0",
                    json={"text": "山里的雾气慢慢散开，露出了青色的石板路。",
                          "voice": "v4_default",
                          "style": "像说书人一样从容地朗读"})
    assert r.status_code == 200, r.text
    tid = r.json()["task_id"]
    deadline, p = time.time() + 90, {}
    while time.time() < deadline:
        p = CLIENT.get(f"/api/tts/progress/{tid}").json()
        if p.get("status") in ("done", "error"):
            break
        time.sleep(0.5)
    assert p.get("status") == "done", p
    assert "说书人" in _last_mock_request()["style"], _last_mock_request()


TESTS = [
    t_engine_config,
    t_engine_config_env_expand,
    t_engine_headers,
    t_style_instruction,
    t_mimo_synth_ok,
    t_mimo_synth_fail_text,
    t_mimo_synth_bad_key,
    t_mimo_synth_missing_key,
    t_mimo_synth_batch,
    t_engine_synth_dispatch,
    t_concat_wav_resample,
    t_e2e_servers,
    t_e2e_voices_local,
    t_e2e_single_tts,
    t_e2e_long_task,
    t_api_emotion_label_forwarded,
    t_api_emotion_header_accepted,
    t_api_emotion_query_accepted,
    t_api_style_field_forwarded,
    t_api_style_header_accepted,
    t_api_style_overrides_emotion,
    t_voices_engine_emotions_isolated,
    t_style_long_task_forwarded,
]


def main():
    proc = _start_mock()
    try:
        for t in TESTS:
            check(t.__name__, t)
    finally:
        proc.kill()
        shutil.rmtree(os.environ["BSAI_TASK_CACHE"], ignore_errors=True)
    print(f"\n===== {len(_PASS)} passed, {len(_FAIL)} failed =====")
    if _FAIL:
        print("failed:", ", ".join(_FAIL))
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
