"""批量/流式接口 + IPv6 + 鉴权 的端到端测试。

用法：
    .\\venv\\Scripts\\python.exe test_batch_api.py

会以子进程方式启动 api_pool.py（继承控制台，避免 ROCm 子进程加载失败），
依次验证：IPv6 健康检查、鉴权拦截、单句、/tts_batch、/tts_batch_stream。
"""

import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

PORT = 9882
KEY = "testkey-123"
HOST4 = f"http://127.0.0.1:{PORT}"
HOST6 = f"http://[::1]:{PORT}"
REF = "SunFLV/Reference_Voices/我的电话号码是13866007742。.wav"
PROMPT = "我的电话号码是13866007742。"
TEXTS = [
    "第一句，测试批量合成。",
    "第二句稍微长一点，用来验证批内并发。",
    "第三句结束。",
]


def call(url, payload=None, key=None, timeout=300):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-API-Key"] = key
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        return resp.status, body, resp.headers, time.perf_counter() - t0


def main():
    cmd = [sys.executable, "api_pool.py", "--pool-size", "2", "--port", str(PORT), "--api-key", KEY]
    print(f"[test] 启动: {' '.join(cmd[1:])}", flush=True)
    proc = subprocess.Popen(cmd, cwd=os.getcwd())
    try:
        ready = False
        for _ in range(40):
            time.sleep(3)
            if proc.poll() is not None:
                print(f"[test] 服务退出 code={proc.returncode}", flush=True)
                return 1
            try:
                st, body, _, _ = call(f"{HOST6}/health")
                print(f"[test] IPv6 READY {json.loads(body)}", flush=True)
                ready = True
                break
            except Exception:
                continue
        if not ready:
            print("[test] 未就绪", flush=True)
            return 1

        base = {"refer_wav_path": REF, "prompt_text": PROMPT, "prompt_language": "zh",
                "text": "单句测试。", "text_language": "zh", "sample_steps": 4}

        # 1) 鉴权：无 key -> 401
        try:
            call(f"{HOST4}/", base)
            print("[test] FAIL 无 key 竟然通过", flush=True)
        except urllib.error.HTTPError as exc:
            print(f"[test] 无 key 被拦截 -> {exc.code}", flush=True)

        # 2) 单句（IPv6 + key）
        st, body, hdr, el = call(f"{HOST6}/", base, key=KEY)
        print(f"[test] 单句 IPv6 status={st} bytes={len(body)} elapsed={el:.2f}s "
              f"sr={hdr.get('X-Sample-Rate')} audio={hdr.get('X-Audio-Seconds')}s", flush=True)

        # 3) 批量
        payload = dict(base)
        payload.pop("text")
        payload["texts"] = TEXTS
        st, body, _, el = call(f"{HOST4}/tts_batch", payload, key=KEY)
        r = json.loads(body)
        wavs = r.get("wavs") or []
        sizes = [len(base64.b64decode(w)) for w in wavs if w]
        print(f"[test] /tts_batch status={st} wall={el:.2f}s n={len(wavs)}/{len(TEXTS)} "
              f"sr={r.get('sample_rate')} wav_bytes={sizes}", flush=True)

        # 4) 流式（NDJSON，按 index 归位）
        req = urllib.request.Request(
            f"{HOST6}/tts_batch_stream",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "X-API-Key": KEY},
            method="POST",
        )
        t0 = time.perf_counter()
        order = []
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line.decode())
                order.append(obj.get("index"))
                print(f"[test] stream line index={obj.get('index')} "
                      f"wav={len(obj.get('wav') or '')} err={obj.get('error')} "
                      f"t={time.perf_counter() - t0:.2f}s", flush=True)
        print(f"[test] /tts_batch_stream 到达序={order} 墙钟={time.perf_counter() - t0:.2f}s "
              f"完整={sorted(n for n in order if n is not None) == list(range(len(TEXTS)))}", flush=True)

        st, body, _, _ = call(f"{HOST4}/health", key=KEY)
        print(f"[test] health={json.loads(body)}", flush=True)
        return 0
    finally:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        except Exception:
            proc.terminate()
        print("[test] 已停止", flush=True)


if __name__ == "__main__":
    sys.exit(main())
