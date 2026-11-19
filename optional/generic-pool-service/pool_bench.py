"""实例池吞吐基准：同一实例池内 串行 vs 并发。

用法：
    .\\venv\\Scripts\\python.exe pool_bench.py --size 3 --requests 6 --steps 4
结果同时打印与写入 output/pool_bench.md
"""

import argparse
import os
import statistics
import time

import numpy as np

from tts_pool import TTSPool, pick_gpu_index

REF = r"SunFLV\Reference_Voices\我的电话号码是13866007742。.wav"
PROMPT = "我的电话号码是13866007742。"
TEXT = "大哥你好，这是实例池并发测试用的句子，看看多路合成到底能不能叠加吞吐。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=4)
    ap.add_argument("--requests", type=int, default=6)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", default="output/pool_bench.md")
    args = ap.parse_args()

    jobs = [{
        "text": args.text, "text_lang": "zh", "ref_audio_path": REF, "prompt_text": PROMPT,
        "prompt_lang": "zh", "sample_steps": args.steps, "streaming_mode": 0, "seed": 42,
    } for _ in range(args.requests)]

    gpu = pick_gpu_index() if args.gpu is None else args.gpu
    with TTSPool(size=args.size, gpu_index=gpu, warm_ref=REF, warm_text="预热一下", warm_prompt=PROMPT) as pool:
        print(f"[bench] pool={args.size} requests={args.requests} steps={args.steps} gpu={gpu}", flush=True)

        t0 = time.perf_counter()
        durs, lat = [], []
        for j in jobs:
            t = time.perf_counter()
            sr, audio = pool.synthesize(**j)
            lat.append(time.perf_counter() - t)
            durs.append(len(np.asarray(audio)) / float(sr))
        wall_serial = time.perf_counter() - t0
        audio_serial = sum(durs)

        t0 = time.perf_counter()
        res = pool.synthesize_many(jobs)
        wall_par = time.perf_counter() - t0
        ok = [r for r in res if not isinstance(r, Exception)]
        err = [r for r in res if isinstance(r, Exception)]
        audio_par = sum(len(np.asarray(a)) / float(s) for s, a in ok)

        speed_serial = audio_serial / wall_serial
        speed_par = audio_par / wall_par
        print(f"[serial] wall={wall_serial:.2f}s audio={audio_serial:.2f}s throughput={speed_serial:.2f}x "
              f"median_latency={statistics.median(lat):.2f}s", flush=True)
        print(f"[concurrent] wall={wall_par:.2f}s audio={audio_par:.2f}s throughput={speed_par:.2f}x "
              f"ok={len(ok)}/{len(jobs)}", flush=True)
        if err:
            print(f"[errors] {len(err)} e.g. {err[0]}", flush=True)
        print(f"[result] speedup={wall_serial / wall_par:.2f}x (ideal {args.size}x) "
              f"throughput_gain={speed_par / speed_serial:.2f}x", flush=True)

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# 实例池并发基准\n\n")
            f.write(f"- 池大小：{args.size} 进程；请求数：{args.requests}；采样步数：{args.steps}\n")
            f.write(f"- 单条音频时长：{durs[0]:.2f} s；GPU={gpu}；启动（含预热）：{pool.startup_seconds:.1f} s\n\n")
            f.write("| 模式 | 总耗时 | 音频总量 | 有效吞吐 | 单请求延迟 |\n|---|---|---|---|---|\n")
            f.write(f"| 串行（1 路） | {wall_serial:.2f} s | {audio_serial:.2f} s | "
                    f"{speed_serial:.2f} 倍实时 | 中位 {statistics.median(lat):.2f} s |\n")
            f.write(f"| 并发（{args.size} 路） | {wall_par:.2f} s | {audio_par:.2f} s | "
                    f"**{speed_par:.2f} 倍实时** | — |\n\n")
            f.write(f"加速比 **{wall_serial / wall_par:.2f}x**（理想 {args.size}x），"
                    f"有效吞吐提升 **{speed_par / speed_serial:.2f}x**。\n")
        print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
