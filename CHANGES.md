# 修改明细

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
