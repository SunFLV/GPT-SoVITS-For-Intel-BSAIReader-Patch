# Generic Pool Service（可选附加件）

本目录是**可选附加件**，不属于主补丁 `bsaireader-gpt-sovits.patch` 的一部分，不会影响补丁的还原校验。

它把本机（RX 9070 XT / ROCm / Windows）在 GPT-SoVITS 上做的「多路并发服务端」抽了出来，特点是：

- **后端中立**：`auto → cuda/ROCm → xpu → mps → dml → cpu`，同一份代码在 CUDA / ROCm / Intel XPU / Apple MPS / DirectML / CPU 都能跑；
- **不依赖主补丁**：只要求目标 GPT-SoVITS 检出里有 `GPT_SoVITS/TTS_infer_pack/TTS.py`（上游 RVC-Boss 或本仓库的 Intel 分支都行）；
- **三层配置 + 探针 + 限流 + 鉴权 + IPv6 双栈**，可直接当常驻服务用。

## 文件与落位

| 文件 | 放到目标检出 | 说明 |
|---|---|---|
| `api_pool.py` | 项目根 | HTTP 服务（`POST /`、`/tts_batch`、`/tts_batch_stream`，探针 `/healthz` `/readyz`） |
| `tts_pool.py` | 项目根 | 进程池（父进程不导入 torch） |
| `tts_worker.py` | 项目根 | 单实例 worker（JSON 行协议） |
| `config.pool_api.example.json` | `config/pool_api.example.json` | 配置模板（复制成 `config/pool_api.json` 使用） |
| `go-pool-api.bat` / `go-pool-api.sh` | 项目根 | Windows / Linux-macOS 启动脚本 |
| `pool_bench.py` | 项目根 | 吞吐基准 |
| `test_api_pool.py` / `test_batch_api.py` / `test_generic_api.py` | 项目根 | 冒烟/回归测试 |
| `instance-pool.md` | `docs/instance-pool.md` | 设计与实测文档 |

全部是**新增文件**，不覆盖主补丁修改过的任何文件。唯一的硬依赖是
`GPT_SoVITS/TTS_infer_pack/device_utils.py`（主补丁里已提供；若用上游代码，把该文件一并放进去即可）。

## 与主补丁「动态接活」的关系

主补丁在 AR 层做 in-model 动态接活（`infer_panel_batch_infer_dynamic_stream`，EOS 腾出的槽位立刻补下一句）。
本附加件把同一件事放到**进程层**：每个 worker 跑完一句立刻从队列拿下一句，空槽不空转，
但不需要改 AR 的 KV/位置编码，风险低得多，也天然跨后端。

本机实测（v5turbo / 4 步 / 40 字文本 / 单条 7.8 s 音频）：

| 指标 | 数值 |
|---|---|
| 单实例串行 | RTF ≈ 0.31（3.2 倍实时） |
| 3 进程并发 | 2.99× 加速，8.8 倍实时 |
| 4 进程并发 | 3.52× 加速，9.4 倍实时（1 小时音频约 6.4 分钟） |
| 显存 | ≈1.75 GB / 实例（16 GB 卡建议 3–4 个） |
| 同进程多线程 + 多 CUDA Stream | **反而变慢 0.62–0.79×**（AR 逐 token 被 GIL 串行化，故必须多进程） |

单实例内也可用 `sample_steps=4` + `cut3` + 串行把首段延迟压到 ~1.1 s。

## 快速开始

```powershell
# Windows
copy config.pool_api.example.json ..\..\config\pool_api.json   # 按需改端口/池大小/API Key
.\go-pool-api.bat
```

```bash
# Linux / macOS
cp config.pool_api.example.json config/pool_api.json
chmod +x go-pool-api.sh && ./go-pool-api.sh
```

最小请求（服务器配了 `default_ref_audio` 后只需 text）：

```bash
curl -X POST http://127.0.0.1:9881/ -H "Content-Type: application/json" \
     -H "X-API-Key: $KEY" -d '{"text":"只传文本。"}' -o out.wav
```

批量 / 流式：

```bash
curl -X POST http://127.0.0.1:9881/tts_batch -H "Content-Type: application/json" \
     -H "X-API-Key: $KEY" -d '{"texts":["第一句。","第二句。"]}'

curl -N -X POST http://127.0.0.1:9881/tts_batch_stream -H "Content-Type: application/json" \
     -H "X-API-Key: $KEY" -d '{"texts":["第一句。","第二句。","第三句。"]}'
```

> 中文请求体请用 UTF-8 文件 + `--data-binary "@body.json"`，PowerShell 直接 `-d '{"text":"中文"}'` 会乱码。

## 配置要点（完整字段见 `config.pool_api.example.json` 与 `instance-pool.md`）

| 字段 | 作用 |
|---|---|
| `pool_size` | 实例数（会被显存上限自动夹紧） |
| `device` / `gpu` | 后端与物理 GPU；`gpu=null` 自动跳过核显 |
| `api_key` | 鉴权；`"auto"` 随机生成；留空则不鉴权（公网务必开） |
| `host` | 默认 `::` = IPv6 双栈；`0.0.0.0` = 仅 IPv4 |
| `default_ref_audio` / `default_prompt_text` | 预置参考音，客户端只传 text |
| `queue_timeout` | 池满时等待秒数，超时返回 **503 + Retry-After** |
| `max_text_chars` | 单句字数上限，超限 400 |
| `cors_origins` | 跨域白名单，`["*"]` 全放开 |
| `warm` / `warm_ref` | 启动预热（每个 worker 首次推理要 ~18 s 内核编译） |

> 注意：worker 是**独立子进程**，父进程不导入 torch；如果父进程先初始化了 HIP 上下文再 spawn，
> ROCm 上会报 `hipErrorInvalidImage`。另外服务必须从**继承控制台**的进程启动（`go-pool-api.*` 即可），
> 用计划任务/重定向脱附控制台启动时同样会触发该错误。
