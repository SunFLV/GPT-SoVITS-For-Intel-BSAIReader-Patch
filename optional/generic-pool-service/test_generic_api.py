"""通用化能力测试：配置优先级、默认参考音、探针、限流、CORS、Request-Id、后端中立。

用法：
    .\\venv\\Scripts\\python.exe test_generic_api.py

会用子进程启动 api_pool.py（继承控制台，避免 ROCm worker 加载失败），
以单实例 + 极短排队超时 + 极小文本上限来覆盖各分支。
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

PORT = 9883
KEY = "generic-test"
BASE = f"http://127.0.0.1:{PORT}"
REF = "SunFLV/Reference_Voices/我的电话号码是13866007742。.wav"
PROMPT = "我的电话号码是13866007742。"


def call(path, payload=None, key=KEY, method=None, headers=None, timeout=300):
    url = BASE + path
    data = json.dumps(payload).encode() if payload is not None else None
    hdrs = {"Content-Type": "application/json"}
    if key:
        hdrs["X-API-Key"] = key
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method or ("POST" if data else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), {k.lower(): v for k, v in exc.headers.items()}


def main():
    cmd = [
        sys.executable, "api_pool.py",
        "--pool-size", "1", "--port", str(PORT), "--api-key", KEY,
        "--default-ref", REF, "--default-prompt", PROMPT, "--default-lang", "zh",
        "--default-steps", "4",
        "--max-text-chars", "8",
        "--queue-timeout", "0.01",
        "--cors", "*",
    ]
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
                st, body, _ = call("/readyz", key=None)
                print(f"[test] /readyz -> {st} {body.decode()}", flush=True)
                ready = st == 200
                break
            except Exception:
                continue
        if not ready:
            print("[test] 未就绪")
            return 1

        st, body, hdr = call("/healthz", key=None)
        print(f"[test] /healthz -> {st} {body.decode()}", flush=True)

        # 1) 默认参考音：只传 text
        st, body, hdr = call("/", {"text": "只有文本。"})
        print(f"[test] 默认参考音(只传 text) -> {st} bytes={len(body)} "
              f"req_id={hdr.get('x-request-id')} sr={hdr.get('x-sample-rate')}", flush=True)
        assert st == 200 and hdr.get("x-request-id"), "默认参考音或 Request-Id 失败"

        # 2) 超长文本被拒（上限 8）
        st, body, _ = call("/", {"text": "这句话明显超过八个字符了，应当被拒绝掉。"})
        print(f"[test] 超长文本 -> {st} {body.decode()[:80]}", flush=True)
        assert st == 400, "max_text_chars 未生效"

        # 3) CORS 预检
        st, body, hdr = call("/", None, method="OPTIONS",
                             headers={"Origin": "https://example.com",
                                      "Access-Control-Request-Method": "POST"})
        print(f"[test] CORS 预检 -> {st} ACAO={hdr.get('access-control-allow-origin')}", flush=True)
        assert st == 204 and hdr.get("access-control-allow-origin") == "*", "CORS 未生效"

        # 4) 单实例 + 极短 queue_timeout -> 并发时至少一个 503
        results = []
        def worker(i):
            results.append(call("/", {"text": "并发测试"})[0])
        ths = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        t0 = time.perf_counter()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        print(f"[test] 3 并发(单实例/queue_timeout=0.01) -> 状态={sorted(results)} "
              f"墙钟={time.perf_counter() - t0:.2f}s", flush=True)
        assert 503 in results, "排队超时未返回 503"

        st, body, _ = call("/health", key=None)
        print(f"[test] /health -> {body.decode()}", flush=True)
        print("[test] 全部通过", flush=True)
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
