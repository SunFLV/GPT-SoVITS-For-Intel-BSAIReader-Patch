"""api_pool.py 冒烟测试：像 go-api.py 那样用 subprocess 启动（继承控制台），再打并发请求。

注意：api_pool / tts_pool 必须从**继承控制台**的进程启动（launcher.py / go-api.py / 普通终端都行）。
用 Start-Process 重定向或计划任务等"脱附控制台"的方式启动时，ROCm 子进程加载会报
hipErrorInvalidImage（本机实测），这不是代码 bug 而是驱动上下文限制。
"""

import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

PORT = 9881
BASE = f"http://127.0.0.1:{PORT}"
REF = r"SunFLV/Reference_Voices/我的电话号码是13866007742。.wav"
PROMPT = "我的电话号码是13866007742。"
TEXT = "多路并发测试，看看实例池能不能同时出音。"
POOL_SIZE = int(os.environ.get("POOL_SIZE_TEST", "0"))  # 0 = 不传，用服务端默认值
ENTRY = os.environ.get("POOL_ENTRY", "python")  # python | bat
CONCURRENCY = 3


def get(path, timeout=5):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def post(payload, tag=""):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + "/", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
        elapsed = time.perf_counter() - t0
        return {
            "tag": tag,
            "status": r.status,
            "elapsed": elapsed,
            "bytes": len(data),
            "sr": r.headers.get("X-Sample-Rate"),
            "audio_s": r.headers.get("X-Audio-Seconds"),
            "x_elapsed": r.headers.get("X-Elapsed"),
        }


def main():
    py = sys.executable
    size_args = ["--pool-size", str(POOL_SIZE)] if POOL_SIZE > 0 else []
    if ENTRY == "bat":
        cmd = ["cmd", "/c", "go-pool-api.bat"] + size_args + ["--port", str(PORT)]
    else:
        cmd = [py, "api_pool.py"] + size_args + ["--port", str(PORT)]
    print(f"[test] entry={ENTRY} cmd={cmd[:3]} pool={'默认' if POOL_SIZE == 0 else POOL_SIZE}", flush=True)
    proc = subprocess.Popen(cmd, cwd=os.getcwd())
    try:
        print(f"[test] 等待服务就绪...", flush=True)
        ready = False
        for _ in range(60):
            time.sleep(3)
            if proc.poll() is not None:
                print(f"[test] 服务进程退出，code={proc.returncode}", flush=True)
                return 1
            try:
                h = get("/health")
                print(f"[test] READY health={h}", flush=True)
                ready = True
                break
            except Exception:
                continue
        if not ready:
            print("[test] 超时未就绪", flush=True)
            return 1

        payload = {
            "refer_wav_path": REF, "prompt_text": PROMPT, "prompt_language": "zh",
            "text": TEXT, "text_language": "zh", "sample_steps": 4,
        }

        single = post(payload, "single")
        print(f"[single] elapsed={single['elapsed']:.2f}s audio={single['audio_s']}s sr={single['sr']} bytes={single['bytes']}", flush=True)

        results = []
        def worker(i):
            try:
                results.append(post(payload, f"c{i}"))
            except Exception as exc:
                results.append({"tag": f"c{i}", "error": repr(exc)})

        t0 = time.perf_counter()
        ths = [threading.Thread(target=worker, args=(i,)) for i in range(CONCURRENCY)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.perf_counter() - t0

        ok = [r for r in results if "error" not in r]
        for r in sorted(ok, key=lambda x: x["tag"]):
            print(f"[{r['tag']}] status={r['status']} elapsed={r['elapsed']:.2f}s "
                  f"audio={r['audio_s']}s sr={r['sr']} bytes={r['bytes']}", flush=True)
        print(f"[concurrent] {len(ok)}/{CONCURRENCY} 成功，墙钟 {wall:.2f}s，"
              f"串行预估 {single['elapsed'] * CONCURRENCY:.2f}s，加速 {single['elapsed'] * CONCURRENCY / wall:.2f}x", flush=True)
        print(f"[health] {get('/health')}", flush=True)
        return 0
    finally:
        # bat 入口下 proc 是 cmd.exe，必须连子进程树一起杀，否则 python 服务会残留并占住端口
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        print("[test] 服务已停止", flush=True)


if __name__ == "__main__":
    sys.exit(main())

