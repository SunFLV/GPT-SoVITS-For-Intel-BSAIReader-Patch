# 实例池：单卡多路并发合成

## 1. 为什么需要"多进程"而不是"多线程"

先做了对照实验（v5turbo / 40 字文本 / 单条 2.1–2.7 s）：

| 方案 | 2 路并发总耗时 | 相对单路 | 结论 |
|---|---|---|---|
| 同进程多线程 + 多 `torch.cuda.Stream` | 6.77 s（每路 6.4–6.8 s） | **0.62×** | 更慢 |
| 多进程（每进程一个 TTS 实例） | 2.40 s（每路 2.38 s） | **1.78×** | 有效 |

原因：AR（GPT）解码是**逐 token 的 Python 循环**，两个线程争 GIL → 没有重叠，只有切换开销。
多进程各自持有 GIL，GPU 侧反而能叠（AR 单步 kernel 很小、利用率低，正好互补）。

## 2. 进程数扫描（每进程一个实例，同卡并发）

| 进程数 | 3 个请求总耗时 | 加速比（理想值） |
|---|---|---|
| 1 | 2.13 s | 1.00× |
| 2 | 2.40 s | 1.78×（2.0） |
| 3 | 2.60 s | 2.46×（3.0） |
| 4 | 2.85 s | **2.99×（4.0）** |
| 5 | 3.69 s | 2.89×（5.0，已饱和） |

**结论：16 GB 卡上 3–4 个实例是甜点**（每实例约 1.75 GB 显存 + 独立 MIOpen 缓存目录）。
5 个开始互相拖慢，因为 GPU 算力见底。

## 3. 池化后的端到端吞吐（`pool_bench.py`）

v5turbo + 4 步 + 40 字文本（单条约 7.8 s 音频）：

| 池大小 | 请求数 | 串行总耗时 | 并发总耗时 | 加速 | **有效吞吐** |
|---|---|---|---|---|---|
| 3 | 6 | 16.05 s | 5.37 s | **2.99×** | **8.76 倍实时** |
| **4（默认）** | 8 | 19.37 s | 5.50 s | **3.52×** | **9.43 倍实时** |

对比单实例的 2.9–3.0 倍实时：**池化把出音吞吐拉到 9–11 倍实时**（1 小时音频约 5.5–7 分钟）。

HTTP 层实测（`api_pool.py`，走 `go-pool-api.bat` 启动）：

| 池大小 | 并发请求 | 单请求 | 并发墙钟 | 串行预估 | 加速 |
|---|---|---|---|---|---|
| 2 | 3 | 2.24 s | 3.46 s | 6.73 s | 1.94× |
| **4（默认）** | 3 | 2.39 s | **2.85 s** | 7.16 s | **2.51×** |

默认池为 **4**（4 × 约 1.75 GB ≈ 7 GB 显存，启动约 43 s 含预热）。

## 4. 结构

```
api_pool.py (FastAPI)             ← HTTP 入口，POST / 兼容 api.py 字段
   └── tts_pool.TTSPool           ← 父进程，不导入 torch；FIFO 分发 + 线程池并发
         └── tts_worker.py × N    ← 独立进程，各加载一份 TTS，stdin/stdout 走 JSON 行协议
pool_bench.py                     ← 吞吐基准（串行 vs 并发）
test_api_pool.py                  ← HTTP 冒烟测试（模拟 go-api.py 的启动方式）
```

协议（每行一个 JSON）：
- worker → 父：`{"type":"ready"}` / `{"type":"warm"}` / `{"type":"warn","msg":..}` / `{"type":"ok","id":N,"sr":48000,"b64":"<int16 PCM base64>"}` / `{"type":"err",...}`
- 父 → worker：`{"id":N,"inputs":{...}}` / `{"type":"shutdown"}`

音频以 int16 + base64 回传（7.6 s / 48 kHz ≈ 720 KB），比 pickle 简单且无版本耦合。

## 5. 用法

### 5.1 一键启动（推荐）

双击或命令行运行根目录的 **`go-pool-api.bat`**（本身就是独立服务端，不依赖 launcher.py）：

```bat
go-pool-api.bat                                  :: 默认 4 实例 / 端口 9881 / 启动预热
go-pool-api.bat --pool-size 3 --port 9882        :: 参数原样透传给 api_pool.py
go-pool-api.bat --pool-size 2 --no-warm          :: 跳过预热（首次请求慢 ~18s/实例）
go-pool-api.bat --help                           :: 看全部参数
```

脚本做了什么：`chcp 65001`、切到脚本目录、用 `venv\Scripts\python.exe` 跑 `api_pool.py`，
设 `PYTHONIOENCODING=utf-8`，出错时 `pause` 保留窗口。启动完成后会打印：

```
============================================================
[api_pool] 就绪：4 个实例 / GPU 1 / 启动 40.5s
[api_pool] 监听 http://0.0.0.0:9881（本机访问 http://127.0.0.1:9881）
[api_pool] 健康检查  curl http://127.0.0.1:9881/health
[api_pool] 接口文档  http://127.0.0.1:9881/docs
============================================================
```

### 5.2 代码里用（不走 HTTP）

```python
from tts_pool import TTSPool
with TTSPool(size=4) as pool:
    sr, audio = pool.synthesize(text="你好", text_lang="zh", ref_audio_path="ref.wav",
                                prompt_text="参考文本", prompt_lang="zh", sample_steps=4)
    results = pool.synthesize_many([dict(...), dict(...), dict(...)])  # 自动并发
```

### 5.3 基准与自测

```powershell
.\venv\Scripts\python.exe pool_bench.py --size 4 --requests 8 --steps 4   # 吞吐基准
.\venv\Scripts\python.exe test_api_pool.py                               # 冒烟（直接起 api_pool.py）
$env:POOL_ENTRY="bat"; .\venv\Scripts\python.exe test_api_pool.py        # 冒烟（走 go-pool-api.bat）
```

### 5.4 HTTP 请求

字段兼容 `api.py`：（`refer_wav_path` / `prompt_text` / `prompt_language` / `text` / `text_language`）

```powershell
curl -X POST http://127.0.0.1:9881/ -H "Content-Type: application/json" -d '{
  "refer_wav_path": "SunFLV/Reference_Voices/我的电话号码是13866007742。.wav",
  "prompt_text": "我的电话号码是13866007742。", "prompt_language": "zh",
  "text": "多路并发测试。", "text_language": "zh", "sample_steps": 4
}'
```

响应为 `audio/wav`，附头部 `X-Sample-Rate` / `X-Elapsed` / `X-Audio-Seconds`；`GET /health` 返回池状态与会话统计。

## 6. 踩过的坑（ROCm 特有，很重要）

1. **父进程不能先初始化 torch 再 spawn**：父进程一旦建立 HIP 上下文，子进程加载权重会报
   `hipErrorInvalidImage`。所以现在父进程完全用 `subprocess` 拉起 `tts_worker.py`，自身不 import torch。
2. **torch 探测子进程也会"污染"接下来几秒**：`pick_gpu_index()` 首次探测后立刻起 worker 同样会失败。
   现在探测结果缓存到 `.gpu_cache.json`（含显存），后续启动直接读缓存，不再探测。**换显卡后删掉该文件即可。**
3. **必须从继承控制台的进程启动**：`launcher.py` / 普通终端 / `go-api.py` 那样 `subprocess.run` 都行；
   用 `Start-Process` 重定向输出或计划任务这类"脱离控制台"的方式启动时，ROCm 子进程加载会报同样的错。
4. **每个 worker 的首次生成要 ~18 s**（内核编译）：`api_pool.py` 默认用
   `SunFLV/Reference_Voices` 下第一个 wav 预热，可用 `--warm-ref` / `--no-warm` 调整。
5. **worker 的 stdin/stdout 强制 UTF-8**：Windows 默认 cp936 会让中文 JSON 解析出错（已修）。
6. **每个 worker 独立 `MIOPEN_*` 缓存目录**，避免多进程并发编译缓存互相踩踏。

## 7. 什么时候不该用池

- **单路、追求最低首段延迟**：单实例 + `cut3` + 串行，首段 1.09 s（池要等 worker 空闲，且预热后单请求 2.2 s 起步）。
- **显存紧张**：每实例 1.75 GB，3 个就 5.3 GB；4 GB 卡只能开 1 个。
- **需要不同参考音频繁切换**：每个 worker 会缓存自己当前的参考音，切换会触发重新编码；
  高频切换场景建议按参考音分组请求，或让每个 worker 固定一个角色。

## 8. 批量 / 流式接口（移植自 BSAIReader 补丁）

请求/响应契约与 `SunFLV/GPT-SoVITS-For-Intel-BSAIReader-Patch` 保持一致，前端可直接对接：

| 端点 | 请求 | 响应 |
|---|---|---|
| `POST /tts_batch` | `/tts` 的全部字段 + `texts: [...]` | `{"sample_rate": 48000, "wavs": ["<b64 wav>", ...]}`，与 `texts` 同序；个别失败时附 `errors: [{index, error}]` |
| `POST /tts_batch_stream` | 同上 | `application/x-ndjson`，每完成一句回一行 `{"index": k, "wav": "<b64>"}`；**到达序 ≠ 句序，调用方按 index 归位** |

```powershell
curl -X POST http://127.0.0.1:9881/tts_batch -H "Content-Type: application/json" -d '{
  "refer_wav_path": "SunFLV/Reference_Voices/我的电话号码是13866007742。.wav",
  "prompt_text": "我的电话号码是13866007742。", "prompt_language": "zh",
  "text_language": "zh", "sample_steps": 4,
  "texts": ["第一句。", "第二句稍微长一点。", "第三句。"]
}'
```

实现差异：补丁是在 **AR 层**做 in-model 动态接活（`infer_panel_batch_infer_dynamic_stream`）；本实例改用 **多进程池**——
每个 worker 跑完一句立刻从队列拿下一句，效果等价（空槽不空转），但不用改 AR 的 KV/位置编码逻辑。
实测（2 worker / 3 句 / 4 步）：`/tts_batch` 2.15 s 出 3 条；`/tts_batch_stream` 三行分别在第 1.33 / 1.63 / 1.85 s 到达。

## 9. IPv6 公网开放

默认绑定 **`::` 双栈**（`IPV6_V6ONLY=0`），同一端口同时接受 IPv6 与 IPv4；`--no-ipv6` 退回仅 IPv4。

启动日志会列出本机全局 IPv6，例如：

```
[api_pool] 监听 [::]:9881（IPv6 双栈，同时接受 IPv4）
[api_pool] 公网   http://[2409:8a1e:be21:6e10:a117:8b93:af72:1e9c]:9881/health   (以太网)
```

放行端口（管理员 CMD，或加 `--open-firewall` 让脚本尝试）：

```bat
netsh advfirewall firewall add rule name="GPT-SoVITS Pool API 9881" dir=in action=allow protocol=TCP localport=9881
```

**安全**：公网暴露必须带鉴权，否则任何人都能烧你的 GPU：

```bat
go-pool-api.bat --api-key auto --open-firewall
```

启动后会打印 `API Key: xxxx`，请求需带 `X-API-Key: xxxx`（或 `?api_key=xxxx`），否则返回 401。
也可以设环境变量 `TTS_API_KEY` 固定 key。`/health` 不鉴权但要保持轻量。

外部验证：手机切 4G/5G（移动网络通常是原生 IPv6）访问 `http://[你的全局IPv6]:9881/health`；
或另一台机器 `curl -6 http://[2409:...]:9881/health`。

注意：
- 家用路由器/光猫的 IPv6 防火墙默认可能拦内网入站，需要在光猫里放行该端口，或临时关闭 IPv6 SPI 防火墙；
- 运营商前缀可能定期变化（动态 IPv6），长期服务建议配 DDNS（如 `dynv6`、`duckdns` 支持 AAAA 记录）；
- 若 ISP 只给 IPv6 但用 NAT66/前缀代理，外部访问会失败，此时只能走内网或 IPv4 端口映射。

## 10. 从 BSAIReader 补丁移植/未移植的内容

| 补丁内容 | 本实例 | 说明 |
|---|---|---|
| `/tts_batch`、`/tts_batch_stream` 契约 | ✅ 已移植 | 用进程池实现动态接活 |
| `device_utils.py`（cuda/xpu/mps/dml/cpu 后端选择） | ✅ 已移植 | 本机走 ROCm（`torch.cuda` 别名）；将来跑 Intel A770 可直接复用 |
| `TextPreprocessor` 短句不再自动补"。" | ✅ 已移植 | 避免首句<4 字时 T2S 立即 EOS、几乎无声 |
| v5turbo 推理支持（models_v5/cache_dit/DiT 分支） | 本实例已有 | 来自上游 `main` 合并，版本更新 |
| AR 层动态接活 `infer_panel_batch_infer_dynamic_stream` | ⛔ 未移植 | 用多进程池替代；改 AR 的 KV/位置编码风险大于收益 |
| mmap 权重加载（`process_ckpt.py`） | ⛔ 未移植 | 本机 47 GB 内存不紧张，且该优化面向 XPU/低内存机型 |
| 默认版本改成 v5turbo / WebUI 版本表 | 本实例已有 | 通过 `tts_infer.yaml` + `weight.json` 已配好 |

## 11. 通用化：配置 / 限流 / 后端 / 跨平台

### 11.1 配置优先级

**命令行 > 环境变量 > 配置文件 > 内置默认值**

配置文件默认读 `config/pool_api.json`（不存在就用默认值），可用 `--config-file` 或环境变量 `POOL_CONFIG_FILE` 指定。
从模板起步：

```powershell
copy config\pool_api.example.json config\pool_api.json
```

字段一览（`config/pool_api.example.json` 里有全部默认值）：

| 字段 | 默认 | 说明 |
|---|---|---|
| `host` / `port` | `"::"` / `9881` | `::` = IPv6 双栈；`0.0.0.0` = 仅 IPv4 |
| `pool_size` | `4` | 实例数（会被显存上限 `max_pool_size` 夹紧） |
| `gpu` | `null` | 物理 GPU 序号；`null` = 自动跳过核显 |
| `device` | `"auto"` | `auto/cuda/cuda:N/xpu/mps/dml/cpu`，交给 `device_utils` 解析 |
| `config` | `GPT_SoVITS/configs/tts_infer.yaml` | TTS 模型配置（相对项目根） |
| `python` | `null` | worker 解释器；`null` = 当前解释器 |
| `api_key` | `""` | 鉴权 key；`"auto"` 自动生成；空 = 不鉴权 |
| `cors_origins` | `[]` | 允许的来源列表，`["*"]` 全放开 |
| `max_text_chars` | `0` | 单句字符上限，`0` = 不限 |
| `queue_timeout` | `120` | 池满时等待空闲实例秒数，超时 **503** |
| `request_timeout` | `900` | 单次合成最长等待秒数 |
| `warm` / `warm_ref` / `warm_text` | `true` / `null` / `"预热一下"` | 启动预热，`warm_ref` 为 `null` 时自动找现成 wav |
| `default_ref_audio` | `null` | 默认参考音，配好后客户端**只需传 text** |
| `default_prompt_text` / `default_prompt_lang` | `null` / `"zh"` | 默认参考音对应的文本与语种 |
| `default_text_lang` | `"zh"` | 默认目标语种 |
| `default_sample_steps` | `null` | 默认采样步数（`null` 用模型默认：v5turbo=8、v4=32） |
| `per_instance_gb` | `2.2` | 每个实例的显存估算，用于池大小上限 |
| `startup_timeout` | `420` | 单实例启动超时秒数 |
| `open_firewall` | `false` | 启动时尝试 netsh 放行端口（需管理员） |

对应环境变量：`POOL_HOST` `POOL_PORT` `POOL_SIZE` `POOL_GPU` `POOL_DEVICE` `POOL_CONFIG`
`POOL_PYTHON` `TTS_API_KEY` `POOL_MAX_TEXT` `POOL_QUEUE_TIMEOUT` `POOL_REQUEST_TIMEOUT`
`POOL_DEFAULT_REF` `POOL_DEFAULT_PROMPT` `POOL_CORS`（逗号分隔）。

### 11.2 探针与限流

| 端点 | 语义 |
|---|---|
| `GET /healthz` | 存活探针，永远 200 |
| `GET /readyz` | 就绪探针：池满员才 200，否则 503（适合负载均衡/容器 readiness） |
| `GET /health` | 详细统计：池大小、存活数、device 列表、served/sentences/rejected/failed、默认值 |
| `queue_timeout` 触发 | 单句返回 `503 + Retry-After: 5`（不再无限等待） |
| `max_text_chars` 触发 | 返回 `400`，错误体带 `request_id` |
| 所有响应 | 带 `X-Request-Id` 头，便于和日志对齐 |

### 11.3 后端中立

worker 用 `TTS_infer_pack/device_utils.resolve_device()` 解析设备，`--gpu N` **只在显式给出时**才设置
`CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES`；`MIOPEN_*` 只在 ROCm 构建下设置。因此同一份代码可用于
CUDA / ROCm / Intel XPU / Apple MPS / DirectML / CPU（`/health` 的 `devices` 会显示实际设备）。

GPU 自动选择顺序：环境变量 `POOL_GPU` > `.gpu_cache.json` 缓存 > 现场探测（跳过名字含 "Graphics" 的核显）。
换显卡后删掉 `.gpu_cache.json` 即可。

### 11.4 路径与跨平台

- 所有路径都相对**项目根**（`tts_pool.py` 所在目录），可以从任意工作目录启动；
- worker 会在启动时 `chdir` 到项目根（TTS 内部有大量相对路径）；
- Windows：`go-pool-api.bat`；Linux/macOS：`./go-pool-api.sh`（自动找 `venv/bin/python` → `.venv/bin/python` → 系统 python，可用 `POOL_PYTHON` 覆盖）。

### 11.5 最小客户端示例（服务器已配默认参考音）

```bash
curl -X POST http://127.0.0.1:9881/ -H "Content-Type: application/json" \
     -H "X-API-Key: $KEY" -d '{"text":"只需文本，参考音用服务器默认的。"}' -o out.wav
```

批量（顺序返回）：

```bash
curl -X POST http://127.0.0.1:9881/tts_batch -H "Content-Type: application/json" \
     -H "X-API-Key: $KEY" -d '{"texts":["第一句。","第二句。"],"sample_steps":4}'
```

流式（NDJSON，按 index 归位）：

```bash
curl -N -X POST http://127.0.0.1:9881/tts_batch_stream -H "Content-Type: application/json" \
     -H "X-API-Key: $KEY" -d '{"texts":["第一句。","第二句。","第三句。"]}'
```

> 注意：用 PowerShell/curl 直接 `-d '{"text":"中文"}'` 可能因编码变成乱码导致 500，请把 JSON 存成 UTF-8 文件再
> `--data-binary "@body.json"`。
