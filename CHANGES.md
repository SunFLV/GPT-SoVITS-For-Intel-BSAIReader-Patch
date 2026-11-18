# 修改明细

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
