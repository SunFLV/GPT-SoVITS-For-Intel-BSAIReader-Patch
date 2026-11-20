# 自报能力（`/api/voices` / `/api/models` / `/api/capabilities`）

新增文件 `bsai_capabilities.py`，并在 `api_v2.py` 的 `APP = FastAPI()` 之后挂载：

| 路由 | 用途 |
|---|---|
| `GET /api/voices` | 音色 + 模型版本，字段与 bsai 代理层 `/api/voices` 完全一致 |
| `GET /api/models` | 本机可用模型版本 |
| `GET /api/capabilities` | 完整清单（模型 / 特性 / 音色明细 / 机器名） |

要点：

- 数据来自 `BSAI_PATCH.json`（补丁声明）与 `BSAI_VOICES.json`（音色清单），
  两者都是安装时生成的新增文件，**不改后端源码**
- 模型列表取「补丁声明 ∩ `tts_infer.yaml` 里真实存在的版本段」，只装一半补丁
  的机器不会误报 v5turbo
- 音色路径统一相对化；绝对路径落在后端目录外时，会在 `bsai/ voices/ ref_audios/`
  里找同名副本并改写，因此清单可以跨机器拷贝
- `/tts`、`/tts_batch`、`/tts_batch_stream` 新增 `voice` 参数：可直接传
  `v4_tomori` 这样的音色名，由清单解析出参考音与 prompt；传真实路径行为不变
- `optional/generic-pool-service/api_pool.py` 同样挂载这三条路由，并按
  「脚本目录 → 上层 → `BSAI_BACKEND_DIR` → cwd」查找 `bsai_capabilities.py`，
  池服务独立部署也能用；`/api/voices` 不要求 `X-API-Key`（只读元数据）

# 修改明细

## 2026-11-19 可选附加件：后端中立的多进程实例池服务（`optional/generic-pool-service/`）

- 新增**可选**附加件，不属于主补丁 `bsaireader-gpt-sovits.patch`，不影响补丁的还原/逐字节校验
- 内容：`api_pool.py` + `tts_pool.py` + `tts_worker.py` + 配置模板 + 启动脚本 + 基准/测试 + 文档
- 定位：把「多路并发服务端」从 in-model 动态接活改成**进程池**实现，不改 AR 的 KV/位置编码，
  因此在 CUDA / ROCm / Intel XPU / Apple MPS / DirectML / CPU 上都能用
- 关键能力：三层配置（CLI > 环境变量 > `config/pool_api.json`）、`default_ref_audio`（客户端只传 text）、
  `/tts_batch`、`/tts_batch_stream`（NDJSON 逐句返回，按 index 归位）、`/healthz` `/readyz`、
  `queue_timeout` 超时 503、`max_text_chars` 超限 400、CORS、API Key、IPv6 双栈
- 本机实测（RX 9070 XT / ROCm / v5turbo / 4 步 / 40 字文本）：
  单实例 RTF ≈ 0.31；3 进程 2.99×（8.8 倍实时）；4 进程 3.52×（9.4 倍实时）；约 1.75 GB 显存/实例
- 反向结论（一并记录，避免再走弯路）：同进程多线程 + 多 CUDA Stream 只有 0.62–0.79×，
  因为 AR 逐 token 的 Python 循环被 GIL 串行化 —— 必须多进程；父进程导入 torch 后再 spawn 会触发
  ROCm `hipErrorInvalidImage`，故父进程不导入 torch、worker 用 subprocess 拉起

## 2026-11-20 可选附加件：v5Turbo INT8 / NVIDIA P106-90（`optional/nvidia-int8/`）

- 新增 `bsai_int8.py`：只在加载 `*_int8.pth` 时把 SoVITS 的 `nn.Linear` 换成
  `Int8Linear`（INT8 权重 + fp16 per-output-channel scale，forward 时反量化），
  兼容 DiT static cache 直接访问 `.weight` 的代码
- 新增 `quantize_v5turbo_int8.py`：把 174 个 Linear 权重离线量化，
  730.9 MiB → 415.3 MiB，前 16 层相对 L2 平均 1.16%
- `optional/generic-pool-service/api_pool.py`：嵌入式 Python 的 `_pth` 不会把脚本
  目录加入 `sys.path`，显式插入自身目录，`runtime_cuda` 下可直接启动
- `tts_worker.py`：`BSAI_INT8_WEIGHTS=1` 时在导入 TTS 前安装 INT8 插件
- 实测（P106-90 3G / Windows / torch 2.5.1+cu121）：加载后 1417 MiB allocated，
  单句峰值 1890 MiB；Pascal sm_61 无 Tensor Core，INT8 只省显存、不提速
- 本机代理新增 `v5Turbo_INT8 NVIDIA` 引擎型服务器（9891），
  长文任务仍由 `server_intel.py` 维护，逐句把合成请求转给该池

## 动态接活 + 逐句流式 `/tts_batch_stream`（长文并行）

- `GPT_SoVITS/AR/models/t2s_model.py` 新增 `infer_panel_batch_infer_dynamic_stream`：
  - 上游批量推理在 EOS 时会把该句从 `y / attn_mask / k_cache / v_cache` 里
    `index_select` 摘掉，所以"空位"天然存在 —— 这里在这个空位上**补下一句**，
    槽位不再空到整批结束（真正的「算好一句接下一句」）
  - 新序列的 KV 走一次单序列 prefill，右补 0 到当前 K 长度并整体 mask（只允许
    看自己前面 `src_len` 列）；位置编码按**每序列自己的步数** gather，否则新
    序列会用到别人的绝对位置，音色/节奏会崩
  - EOS 那一刻 `attn_mask` 比 cache 多一列（上游给下一步的占位），插入时以
    cache 长度为准裁掉，下一步主循环会重新 pad 回来
  - generator：每生成完一句就 `yield (原始句号, tokens, 有效长度)`
- `TTS_infer_pack/TTS.py`：`dynamic_batch` 输入下逐句解码（DiT/声码器）并逐句
  `yield`；`texts` 分支在动态模式下不做 `to_batch`（需要保持"逐句"才能接活）
- `api_v2.py` 新增 `POST /tts_batch_stream`：NDJSON，一句一行
  `{"index": k, "wav": "<base64>"}`。**完成顺序不等于句序**（实测到达序
  8,6,4,1,10,12…），调用方必须按 `index` 归位 —— 这是"顺序不会错"的保证
- 实测（A770 / v5turbo，16 句长短混排）：总 136.3 s，16/16 句齐全；第 1 句
  19.4 s 就可在点读里听（静态批要等整批 60 s 以上）；音频时长与句长相关性 0.94
- 兜底：`dynamic_batch` 报错/老后端没有该端点 → 服务端退回 `/tts_batch`
  静态批 → 再不行逐句；个别句子静音只补那一句

## 逐句批量推理 `/tts_batch`（长文并行）

- 新增 `api_v2.py` 的 `POST /tts_batch`：请求体 = `/tts` 的全部参数 + `texts: [...]`，
  返回 `{"sample_rate", "wavs": [<base64 wav>, ...]}`，顺序与 `texts` 一致
- `TTS_infer_pack/TTS.py`：
  - `run()` 新增输入 `texts`：跳过 `preprocess()` 的按标点切分，**每个 text 就是一段**，
    语音前端（BERT/CNHuBERT）逐句跑、模型推理把这几句凑成一个 batch
  - 新增输入 `return_items`：不拼接，按输入顺序逐句做后处理（限幅 + 句间静音）后
    以列表返回，调用方拿到的仍是逐句 WAV，点读偏移不受影响
- 为什么必须这么做：`/tts` 的推理在事件循环里同步执行，**请求之间严格串行**，
  一句一请求时 `batch_size` 只作用于“这一句被切成几段”，短句根本切不出第二段
- A770 / v5turbo 实测（同一批文本）：逐句 12.1 s/句；4 句一批 8.2 s/句（1.5×）；
  8 句一批 5.6 s/句（2.1×）；12/16 句不再提升（6.5 s/句），只增加显存峰值
- 与客户端约定：批量失败（老后端 404、接口报错）时客户端自动退回逐句；
  个别句子静音时只补那几句

## mmap 权重加载（省内存）

- `process_ckpt.py`：`load_sovits_new()` 新增 `device` / `use_mmap` 参数，标准 ZIP 检查点用 `mmap=True` 直接映射到指定设备；旧版非 ZIP 格式无法 mmap，保持原 CPU 加载；mmap 抛异常时自动回退
- `TTS_infer_pack/TTS.py`：
  - `init_vits_weights()` 调用 `load_sovits_new(..., device=xpu, use_mmap=True)`
  - `init_t2s_weights()` 用 `torch.load(..., map_location=device, mmap=True)`，失败回退
  - `init_vocoder()` 同样改为 mmap 直连设备，失败回退 CPU
- 实测：SoVITS 权重文件 731 MB，传统 CPU 加载进程私有内存 2088 MB、工作集 1020 MB；mmap+XPU 为 1585 MB / 288 MB
- 代价：首次模型加载从约 5 s 增至约 27 s（mmap 页首次载入设备）；换来的低内存驻留适合内存受限机器

## v5Turbo 推理支持

- 新增 `GPT_SoVITS/module/models_v5.py`（`CFMV5`、`synthesize_v5_mel`、v5 常量）
- 新增 `GPT_SoVITS/module/cache_dit.py`
- 更新 `GPT_SoVITS/module/models.py`（CFM/DBCache、SynthesizerTrnV3 的 v5 分支）
- 更新 `GPT_SoVITS/f5_tts/model/backbones/dit.py`、`f5_tts/model/__init__.py`
- 更新 `GPT_SoVITS/module/modules.py`、`GPT_SoVITS/process_ckpt.py`（v5turbo 签名）
- `TTS_infer_pack/TTS.py`：
  - 导入 `V5_VERSIONS` / `synthesize_v5_mel`
  - 注册 v5dev/v5turbo 默认配置；版本断言与 v3/v4 集合纳入 v5
  - 声码器路径按版本选择 `gsv-v4-pretrained` / `gsv-v5-pretrained`
  - v5turbo 默认 4 步，v4 默认 32 步
  - `using_vocoder_synthesis()` 及 batched 版本加入 v5 mel 合成分支
  - **`save_configs()` 保留顶层 `version`**：否则保存后 YAML 丢失版本，下次加载回落到 v2（实测导致 v5 跑成 v2 的严重问题）
- `configs/tts_infer.yaml`：顶层与 custom 版本 `v5turbo`，新增 `v5turbo:` 预设段
- `webui.py`：默认 v5turbo，版本单选提供 v4/v5turbo，v5 权重路径与索引映射
- 两个 `inference_webui*.py`：默认 v5turbo，`v3v4set` 纳入 v5turbo，LoRA 底模检查支持 v5turbo
- `config.py`：默认 pretrainedSoVITS/GPT 路径改为 v5turbo 对应文件

## 此前已包含

- 后端通用化：`device_utils.py`、`resolve_device()`、安全 `empty_device_cache()`、IPEX/XPU 保护、GPU 探测支持 CUDA/XPU/MPS/DML
- 短文本静音修复：`TextPreprocessor.py` 不再给短句自动补标点
- 切换音色时彻底清空 prompt 缓存（含 `aux_ref_audio_paths` / `raw_sr`）

## 排除项

`weight.json`、角色权重、`user.dict`、生成音频、`runtime/` 环境变化与 `.pyc`，详见 `MANIFEST.json`。
