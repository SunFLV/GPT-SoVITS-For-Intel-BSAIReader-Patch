# 修改明细

## v5Turbo 推理支持（本次新增）

- 从上游 `RVC-Boss/GPT-SoVITS` main 分支移植 v5 所需核心实现：
  - 新增 `GPT_SoVITS/module/models_v5.py`（`CFMV5`、`synthesize_v5_mel`、v5 常量）
  - 新增 `GPT_SoVITS/module/cache_dit.py`
  - 更新 `GPT_SoVITS/module/models.py`（CFM/DBCache、SynthesizerTrnV3 的 v5 分支）
  - 更新 `GPT_SoVITS/f5_tts/model/backbones/dit.py` 与 `f5_tts/model/__init__.py`（upstream DiT）
  - 更新 `GPT_SoVITS/module/modules.py` 与 `GPT_SoVITS/process_ckpt.py`（v5turbo 权重签名 `6bc5cb2a...`）
- `TTS_infer_pack/TTS.py`：
  - 导入 `V5_VERSIONS` / `synthesize_v5_mel`，注册 `v5dev` / `v5turbo` 默认配置
  - 版本断言、模型版本识别、v3/v4 集合判断纳入 v5
  - 声码器路径按版本选择 `gsv-v4-pretrained` 或 `gsv-v5-pretrained`
  - `sample_steps` 对 v5turbo 默认 4，v5dev/其他默认 32
  - `using_vocoder_synthesis()` 与 batched 版本加入 v5 mel 合成分支
- `configs/tts_infer.yaml`：顶层与 custom 版本改为 `v5turbo`，权重路径指向 `s2Gv5turbo.pth`，新增 `v5turbo:` 预设段
- `webui.py`：默认 v5turbo，版本单选提供 v4/v5turbo，新增 v5 权重路径与版本索引映射
- 两个 `inference_webui*.py`：默认 v5turbo，`v3v4set` 纳入 v5turbo，LoRA 底模缺失检查支持 v5turbo 路径
- `config.py`：默认 pretrainedSoVITS/GPT 路径改为 v5turbo 对应文件

## 此前已包含

- 后端通用化：`device_utils.py`、`resolve_device()`、安全的 `empty_device_cache()`、IPEX/XPU 保护、GPU 探测支持 CUDA/XPU/MPS/DML
- 短文本静音修复：`TextPreprocessor.py` 不再给短句自动补标点
- 切换音色时彻底清空 prompt 缓存，包含 `aux_ref_audio_paths` / `raw_sr`

## 仍需用户准备的底模

- `GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5turbo.pth`
- `GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth`

## 排除项

`weight.json`、角色权重、`user.dict`、生成音频、`runtime/` 环境变化与 `.pyc`，详见 `MANIFEST.json`。
