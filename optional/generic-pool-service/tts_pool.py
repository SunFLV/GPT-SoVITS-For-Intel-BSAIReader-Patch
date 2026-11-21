"""多进程实例池：单卡多路并发合成（后端中立，可从任意工作目录导入）。

背景（RX 9070 XT / ROCm / v5turbo / 40 字文本单次 2.1–2.6 s）：
- 同进程多线程 + 多 CUDA Stream：**变慢 0.62–0.79x** —— AR 逐 token 的 Python 循环被 GIL 串行化；
- 多进程（每进程一个实例）：2 路 1.78x、3 路 2.46x、4 路 2.99x、5 路 2.89x（开始饱和）。

实现要点：
- 父进程**不导入 torch**：worker 用 `subprocess` 拉起独立进程（stdin/stdout JSON 行协议），
  否则父进程已初始化 HIP 上下文时，子进程加载权重会报 hipErrorInvalidImage（ROCm 实测）；
- 后端/显卡由 worker 侧解析（`--device` 交给 device_utils，`--gpu` 才设可见性环境变量），
  因此 CUDA / ROCm / XPU / MPS / CPU 都能用；
- 路径全部相对**本文件所在目录**（项目根），从任意 cwd 启动都行。

用法：
    from tts_pool import TTSPool
    with TTSPool(size=4) as pool:
        sr, audio = pool.synthesize(text="你好", text_lang="zh", ref_audio_path="ref.wav",
                                    prompt_text="参考文本", prompt_lang="zh", sample_steps=4)
        results = pool.synthesize_many([dict(...), dict(...)])
"""

import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT = os.path.join(PROJECT_ROOT, "tts_worker.py")
DEFAULT_CONFIG = os.path.join("GPT_SoVITS", "configs", "tts_infer.yaml")
GPU_CACHE = os.path.join(PROJECT_ROOT, ".gpu_cache.json")

PASS_KEYS = (
    "text", "text_lang", "ref_audio_path", "aux_ref_audio_paths", "prompt_text", "prompt_lang",
    "top_k", "top_p", "temperature", "text_split_method", "batch_size", "batch_threshold",
    "speed_factor", "split_bucket", "return_fragment", "fragment_interval", "seed",
    "parallel_infer", "use_cuda_graph", "use_flash_attention", "repetition_penalty",
    "sample_steps", "cfg_rate", "cache_dit", "streaming_mode",
)


class PoolBusy(RuntimeError):
    """池内没有空闲 worker（用于返回 503 而不是无限等待）。"""


# ------------------------------------------------------------------ 显卡探测
def _probe_gpus():
    """一次性子进程探测显卡（编号/名称/显存 GB），父进程不长期持有 torch 上下文。"""
    code = (
        "import json,torch;"
        "print(json.dumps([[i, torch.cuda.get_device_name(i),"
        " torch.cuda.get_device_properties(i).total_memory/1024**3]"
        " for i in range(torch.cuda.device_count())]))"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        lines = (out.stdout or "").strip().splitlines()
        data = json.loads(lines[-1]) if lines else []
        return [(int(i), str(n), float(m)) for i, n, m in data]
    except Exception:
        return []


def pick_gpu_index(use_cache=True):
    """选择要用的物理 GPU 序号；返回 None 表示让 torch 自己决定（CPU/单卡/非 CUDA 后端）。

    优先级：环境变量 POOL_GPU > 缓存文件 > 现场探测。
    探测规则：跳过名字里带 "Graphics" 的设备（通常是 AMD/Intel 核显），选第一个独立显卡。
    """
    env_gpu = os.environ.get("POOL_GPU")
    if env_gpu not in (None, ""):
        try:
            return int(env_gpu)
        except ValueError:
            pass
    if use_cache and os.path.exists(GPU_CACHE):
        try:
            with open(GPU_CACHE, encoding="utf-8") as f:
                return int(json.load(f)["index"])
        except Exception:
            pass
    gpus = _probe_gpus()
    if not gpus:
        return None
    idx, total_gb = gpus[0][0], gpus[0][2]
    for i, name, mem in gpus:
        if "graphics" not in name.lower():
            idx, total_gb = i, mem
            break
    try:
        with open(GPU_CACHE, "w", encoding="utf-8") as f:
            json.dump({"index": idx, "total_gb": total_gb, "gpus": gpus}, f, ensure_ascii=False)
    except Exception:
        pass
    return idx


def max_pool_size(gpu_index=None, per_instance_gb=2.2, safety=0.85):
    """按显存估算安全池大小；优先读缓存，避免再起 torch 探测子进程。"""
    total_gb = None
    if os.path.exists(GPU_CACHE):
        try:
            with open(GPU_CACHE, encoding="utf-8") as f:
                total_gb = float(json.load(f)["total_gb"])
        except Exception:
            total_gb = None
    if total_gb is None:
        return 3  # 探测不到时保守值
    return max(1, int(total_gb * safety // per_instance_gb))


# ------------------------------------------------------------------ worker 句柄
class _Worker:
    """一个 worker 子进程 + 它的 stdout 行读取线程。"""

    def __init__(self, proc):
        self.proc = proc
        self.pid = proc.pid
        self.device = "?"
        self.lines = queue.Queue()
        self.last_log = ""
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("{"):
                    self.last_log = line  # torch/warning 之类的普通输出
                    continue
                try:
                    json.loads(line)
                except Exception:
                    self.last_log = line
                    continue
                self.lines.put(line)
        except Exception:
            pass
        self.lines.put(None)  # EOF 哨兵

    def send(self, obj):
        self.proc.stdin.write(json.dumps(obj, ensure_ascii=True) + "\n")
        self.proc.stdin.flush()

    def recv(self, timeout=600):
        line = self.lines.get(timeout=timeout)
        if line is None:
            raise RuntimeError(f"worker {self.pid} 已退出(returncode={self.proc.poll()})：{self.last_log}")
        return json.loads(line)

    def alive(self):
        return self.proc.poll() is None

    def close(self):
        try:
            self.send({"type": "shutdown"})
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass


# ------------------------------------------------------------------ 池
class TTSPool:
    """N 个独立进程的 TTS 实例池；线程安全（内部 FIFO 分发 worker）。"""

    def __init__(self, size=4, config_path=None, gpu=None, device="auto", python=None,
                 warm_ref=None, warm_text="预热一下", warm_prompt=None, per_instance_gb=2.2,
                 spawn_timeout=420, request_timeout=900, verbose=True):
        self.size = max(1, int(size))
        self.config_path = config_path or DEFAULT_CONFIG
        if not os.path.isabs(self.config_path):
            self.config_path = os.path.join(PROJECT_ROOT, self.config_path)
        self.gpu = pick_gpu_index() if gpu is None and gpu is not False else gpu
        self.device = device
        self.python = python or os.environ.get("POOL_PYTHON") or sys.executable
        self.per_instance_gb = per_instance_gb
        self.spawn_timeout = spawn_timeout
        self.request_timeout = request_timeout
        self.verbose = verbose
        self._workers = queue.Queue()
        self._all = []
        self._closed = False
        self._jid = 0
        self.startup_seconds = 0.0
        # worker 自愈：记住启动参数，后台线程在成员意外退出时补回。
        # OOM 等系统性问题会让新 worker 立刻再死，因此带退避，避免重启风暴。
        self._warm_ref = warm_ref
        self._warm_text = warm_text
        self._warm_prompt = warm_prompt
        self._heal_lock = threading.Lock()
        self._healing = threading.Event()
        self._heal_thread = None
        self._start(warm_ref, warm_text, warm_prompt)
        self._heal_thread = threading.Thread(target=self._heal_loop, daemon=True, name="pool-healer")
        self._heal_thread.start()

    # ---------- 启动 ----------
    def _spawn(self, warm_ref, warm_text, warm_prompt):
        cmd = [self.python, WORKER_SCRIPT, "--config", self.config_path, "--device", self.device]
        if self.gpu is not None and int(self.gpu) >= 0:
            cmd += ["--gpu", str(int(self.gpu))]
        if warm_ref:
            cmd += ["--warm-ref", warm_ref, "--warm-text", warm_text or "预热一下",
                    "--warm-prompt", warm_prompt or warm_text or "预热一下"]
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
        )
        return _Worker(proc)

    def _start(self, warm_ref, warm_text, warm_prompt):
        t0 = time.time()
        await_ready = self._make_ready_waiter()

        # 先全部并发拉起（错峰 0.3s），再逐个等待就绪；失败的重启重试
        pending = []
        for _ in range(self.size):
            pending.append(self._spawn(warm_ref, warm_text, warm_prompt))
            time.sleep(0.3)

        for i, w in enumerate(pending):
            ok_worker = None
            for attempt in range(3):
                try:
                    ok = await_ready(w)
                except Exception as exc:
                    if self.verbose:
                        print(f"[pool] 成员 {i} 启动异常(第{attempt + 1}次): {exc}", flush=True)
                    ok = False
                if ok:
                    ok_worker = w
                    break
                try:
                    w.close()
                except Exception:
                    pass
                time.sleep(1.5)
                w = self._spawn(warm_ref, warm_text, warm_prompt)
            if ok_worker is None:
                raise RuntimeError(f"第 {i} 个池成员启动失败（已重试 3 次）")
            self._all.append(ok_worker)
            self._workers.put(ok_worker)

        self.startup_seconds = time.time() - t0
        if self.verbose:
            devices = {w.device for w in self._all}
            print(f"[pool] {self.size} 个实例就绪，用时 {self.startup_seconds:.1f}s，"
                  f"device={','.join(sorted(devices))} gpu={self.gpu}", flush=True)

    # ---------- worker 自愈 ----------
    def _heal_loop(self):
        """后台守护：成员意外退出时清理并补回，避免一次 OOM 让池永久失效。

        触发来源：synthesize() 检测到死 worker 时会 set _healing；此外每 10 s
        兜底巡检一次（覆盖「worker 死时无请求在途」的情况）。
        补员失败会指数退避（30 s → 60 s → 120 s，上限 300 s），防止系统仍在
        OOM 时反复拉起 worker 造成重启风暴；成功后退避清零。
        """
        backoff = 30
        while not self._closed:
            self._healing.wait(10)
            self._healing.clear()
            if self._closed:
                break
            try:
                healed = self._replenish()
            except Exception as exc:
                if self.verbose:
                    print(f"[pool] 自愈异常: {exc}", flush=True)
                healed = False
            if healed:
                backoff = 30
            else:
                if self.verbose:
                    print(f"[pool] 补员失败，{backoff}s 后重试", flush=True)
                self._healing.wait(backoff)
                self._healing.clear()
                backoff = min(backoff * 2, 300)

    def _replenish(self):
        """清理死 worker 并补到 self.size；返回本次是否做了有效动作。"""
        with self._heal_lock:
            if self._closed:
                return False
            # 1) 从队列剔除已死的 worker（否则它们占着名额，请求会拿到死句柄）
            live = []
            dead = 0
            while True:
                try:
                    w = self._workers.get_nowait()
                except queue.Empty:
                    break
                if w.alive():
                    live.append(w)
                else:
                    dead += 1
            for w in live:
                self._workers.put(w)
            # 2) 同步 _all 列表
            self._all = [w for w in self._all if w.alive()]
            need = self.size - len(self._all)
            if need <= 0 and dead == 0:
                return False
            if self.verbose:
                print(f"[pool] 检测到 {dead} 个成员退出，补员 {need} 个", flush=True)
            # 3) 逐个补回：复用 _spawn + await_ready 就绪判定
            for i in range(need):
                w = self._spawn(self._warm_ref, self._warm_text, self._warm_prompt)
                try:
                    await_ready = self._make_ready_waiter()
                    ok = await_ready(w)
                except Exception as exc:
                    ok = False
                    if self.verbose:
                        print(f"[pool] 补员就绪异常: {exc}", flush=True)
                if not ok:
                    try:
                        w.close()
                    except Exception:
                        pass
                    return False
                self._all.append(w)
                self._workers.put(w)
                if self.verbose:
                    print(f"[pool] 补员 {i + 1}/{need} 就绪 (device={w.device})", flush=True)
            return True

    def _make_ready_waiter(self):
        """把 _start 里的就绪判定抽成可复用函数（补员时复用同一套逻辑）。"""
        expect_warm = bool(self._warm_ref and os.path.exists(self._warm_ref))

        def await_ready(w):
            deadline = time.time() + self.spawn_timeout
            while True:
                msg = w.recv(timeout=max(1.0, deadline - time.time()))
                kind = msg.get("type")
                if kind == "fatal":
                    if self.verbose:
                        print(f"[pool] 成员加载失败: {msg.get('msg')}", flush=True)
                    return False
                if kind == "warn":
                    if self.verbose:
                        print(f"[pool] {msg.get('msg')}", flush=True)
                    continue
                if kind == "ready":
                    w.device = str(msg.get("device", "?"))
                    if not expect_warm:
                        return True
                if kind == "warm":
                    return True
        return await_ready

    # ---------- 生命周期 ----------
    def close(self):
        if self._closed:
            return
        self._closed = True
        self._healing.set()          # 唤醒守护线程让它看到 _closed 退出
        for w in self._all:
            try:
                w.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def alive(self):
        return sum(1 for w in self._all if w.alive())

    @property
    def devices(self):
        return sorted({w.device for w in self._all})

    # ---------- 请求 ----------
    def synthesize(self, wait_timeout=None, **inputs):
        """同步合成一条，返回 (sr, np.int16 数组)。

        wait_timeout: 等待空闲 worker 的最长时间；超时抛 PoolBusy（HTTP 层映射为 503）。
        """
        if self._closed:
            raise RuntimeError("pool closed")
        payload = {k: v for k, v in inputs.items() if k in PASS_KEYS and v is not None}
        try:
            w = self._workers.get(timeout=wait_timeout)
        except queue.Empty as exc:
            raise PoolBusy(f"池内 {self.size} 个实例都在忙，等待 {wait_timeout}s 超时") from exc
        self._jid += 1
        jid = self._jid
        try:
            w.send({"id": jid, "inputs": payload})
            deadline = time.time() + self.request_timeout
            while True:
                msg = w.recv(timeout=max(1.0, deadline - time.time()))
                if msg.get("type") == "ok" and msg.get("id") == jid:
                    audio = np.frombuffer(base64.b64decode(msg["b64"]), dtype=np.int16)
                    return int(msg["sr"]), audio
                if msg.get("type") == "err" and msg.get("id") == jid:
                    raise RuntimeError(f"worker 合成失败: {msg.get('msg')}")
        except (queue.Empty, OSError, ValueError) as exc:
            raise RuntimeError(f"worker 通信失败: {exc!r}") from exc
        finally:
            if w.alive():
                self._workers.put(w)
            else:
                # worker 在请求中死亡（OOM/CUDA 错误等）：唤醒后台守护补员，
                # 否则池会永久少一个实例，后续请求全部卡在 _workers.get() 超时。
                self._healing.set()

    def synthesize_many(self, jobs, max_workers=None, wait_timeout=None):
        """并发合成多条，返回同序结果（失败位置为 Exception）。"""
        from concurrent.futures import ThreadPoolExecutor

        results = [None] * len(jobs)
        n = max_workers or self.size

        def _one(i):
            try:
                results[i] = self.synthesize(wait_timeout=wait_timeout, **jobs[i])
            except Exception as exc:
                results[i] = exc

        with ThreadPoolExecutor(max_workers=n) as ex:
            list(ex.map(_one, range(len(jobs))))
        return results
