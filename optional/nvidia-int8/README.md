# v5Turbo INT8 / NVIDIA（P106-90 3G）附加件

把 v5Turbo 的 SoVITS Linear 权重量化成 INT8，在 3 GiB 显存的 Pascal 卡上跑起来。
不替换原来的 `runtime`（Intel XPU/IPEX），两条后端可以同时存在。

## 为什么是「权重量化」而不是 CUDA INT8 GEMM

P106-90 是 GP106 / sm_61，没有 Tensor Core；PyTorch 的
`torch.quantization.quantize_dynamic` 在 CUDA 上主要走 FBGEMM/CPU 路径，
直接搬到 CUDA 要么报 `quantized::linear_dynamic` 不支持，要么比 fp16 更慢。
所以这里做的是：

- 离线把 `nn.Linear` 的二维权重按输出通道对称量化成 `int8`；
- 推理时在 `Int8Linear.forward` 里按当前激活 dtype 反量化，再走原来的 fp16 GEMM；
- 省的是**常驻显存**，不是算力。Pascal 上不要期待 INT8 加速。

`nn.Embedding`、`Conv1d`、归一化层保持 fp16，避免音质和 conv 路径出问题。

## 文件

| 文件 | 放到后端根目录 | 作用 |
|---|---|---|
| `bsai_int8.py` | 根目录 | `Int8Linear` + `TTS.init_vits_weights` 补丁 |
| `quantize_v5turbo_int8.py` | 根目录 | 生成 `s2Gv5turbo_int8.pth` |
| `tts_infer_v5turbo_int8.yaml` | `GPT_SoVITS/configs/` | CUDA / INT8 / 4 步配置 |
| `pool_api_nvidia_int8.json` | `config/` | 9891 端口、单实例、电话参考音 |
| `start_nvidia_int8.bat` | 根目录 | 启动池服务 |

另需把通用池服务的 `api_pool.py`、`tts_pool.py`、`tts_worker.py` 放到后端根目录
（`optional/generic-pool-service/` 里已有）。

## 准备 CUDA 运行时

原来的 `runtime` 是 `torch 2.7.0+xpu`，不能跑 CUDA。复制一份并换成 CUDA 版：

```powershell
# 只复制运行时，不复制 XPU 的 torch/ipex 目录（后面会重装）
robocopy runtime runtime_cuda /E /XD runtime\Lib\site-packages\torch `
  runtime\Lib\site-packages\torchaudio runtime\Lib\site-packages\torchvision `
  runtime\Lib\site-packages\intel_extension_for_pytorch `
  runtime\Lib\site-packages\pytorch_triton_xpu runtime\Lib\site-packages\triton

runtime_cuda\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu121 `
  --no-deps --force-reinstall torch==2.5.1+cu121 torchaudio==2.5.1+cu121
```

验证：

```powershell
runtime_cuda\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2.5.1+cu121 True NVIDIA P106-090
```

> 驱动 566.14 自带 CUDA 12.7，向下兼容 cu121 运行时。P106 是 sm_61，
> PyTorch 2.5.1 的 cu121 二进制仍包含 sm_61 kernel。

## 量化与启动

```powershell
# 一次即可；输出 415 MiB 左右
runtime\python.exe quantize_v5turbo_int8.py --verify

# 启动 9891 池（会预热一次）
start_nvidia_int8.bat
```

`start_nvidia_int8.bat` 会设置 `BSAI_INT8_WEIGHTS=1`、`PYTHONPATH`、
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，然后用
`runtime_cuda\python.exe api_pool.py --config-file config\pool_api_nvidia_int8.json`
启动。池是**单实例**（3 GiB 放不下第二份），父进程不导入 torch。

## 接入本机代理

`server/config_intel.yaml` 里新增一条引擎型服务器：

```yaml
  - name: "v5Turbo_INT8 NVIDIA"
    url: "http://127.0.0.1:9891"
    desc: "P106-90 3G · CUDA / v5Turbo INT8 权重（省显存）"
    engine:
      api_key: "bsai-893c84f29fe484cab774d999d6caa1a4"
      default_voice: "v4_phone"
      voice_map:
        v4_sunflv: v4_phone
```

长文任务、进度、点读、断点续传仍在跑 `server_intel.py` 的那台机器上；
每一句的合成请求转发到 9891。前端下拉框里选它即可，不需要直连 P106 那台。

## 实测（P106-90 3G / Windows / WDDM）

| 指标 | 数值 |
|---|---|
| 量化文件 | 730.9 MiB → **415.3 MiB** |
| 量化误差（前 16 个 Linear 的相对 L2） | 平均 1.16%，最大 1.58% |
| 模型加载 | 约 23 s（冷启动） |
| 加载后显存 | allocated 1417 MiB / reserved 2074 MiB |
| 单句峰值 | max allocated 1890 MiB |
| 7.0 s 音频 | 约 33 s（首次含预热/内核初始化；后续会快一些） |

3 GiB 卡上不要开 `pool_size > 1`，也不要开批量并行；池服务默认逐句处理。

## 对照基准（40 字中文 / seed 42 / 4 步 / 预热后 3 次均值）

| 配置 | 平均耗时 | RTF | 显存 alloc（加载后） | reserved 峰值 |
|---|---|---|---|---|
| **P106-90 INT8**（CUDA / cu121） | **8.56 s** | 1.04 | 1417 MiB | 2074 MiB |
| P106-90 fp16（CUDA / cu121） | 8.30 s（−3%） | 1.01 | 1741 MiB | **2564 MiB** |
| **A770 原版 fp16**（XPU / torch 2.7） | 18.57 s | 2.25 | 1741 MiB | 1982 MiB |
| A770 INT8（XPU / torch 2.7） | 18.51 s（±0.3%，持平） | 2.25 | 1417 MiB | 2074 MiB |

结论：

- **A770 上 INT8 没有速度收益**（18.51 vs 18.57 s，噪声范围），只有 ~326 MiB
  显存节省 —— A770 有 15.5 GiB，显存不是瓶颈，**保持原版 fp16**；
- P106 上 fp16 略快 3%，但 reserved 峰值 2564 MiB 已贴近 3 GiB 上限，
  **保持 INT8** 换取余量；
- 值得注意：**P106 反而比 A770 快一倍**（8.3~8.6 s vs 18.5 s）—— 之前的
  IPEX 分析已经定位到 A770 的瓶颈是 CPU 侧 kernel launch（~84 µs/次，
  CUDA 只有 ~3 µs），这张便宜的老卡在「大量小算子」的 v5Turbo 工作负载上
  总时间反而占优。

基准脚本：`server/bench_int8_cards.py`（同一文本/seed/参考音，预热一次后
计 3 次均值，输出 JSON SUMMARY 行）。

## 短句静音

`v4_phone` 这段 5 秒电话参考音在 P106 上对 **3~8 个字**的句子会直接返回
0.24 s、RMS≈0.002 的静音（同样的句子用 `v4_default` / `v4_tomori` 正常）。
代理层已经处理：

- 引擎合成检测到静音后，把短句核心重复一遍再试：
  `最终验证。` → `最终验证，最终验证。`
- `名字说` 这种 2~4 字的说话人前缀若仍静音，自动跳过前缀，只合成正文，
  不会让整条 `/api/tts` 返回 502
- 长文分句本身会把 ≤6 字的碎片并到相邻句，正常小说正文很少触发

池服务的 `/tts_batch` 不做改写，所以直接调用池 API 时短句仍可能静音；
走本机代理的引擎模式才有上述兜底。
