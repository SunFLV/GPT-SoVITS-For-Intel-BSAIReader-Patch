# GPT-SoVITS-For-Intel — BSAIReader 通用后端 + v5Turbo 补丁

由未修改的 `GPT-SoVITS-For-Intel.zip` 与已修改目录逐文件比对生成，包含后端通用化、默认 v5Turbo 推理支持，以及此前的短句静音/缓存修复。

## 默认模型版本：v5Turbo

- 顶层 `version: v5turbo`，`custom.version: v5turbo`
- `custom.vits_weights_path` 指向 `gsv-v5-pretrained/s2Gv5turbo.pth`
- `custom.t2s_weights_path` 仍为 `s1v3.ckpt`
- WebUI 版本选择提供 `v4` 与 `v5turbo`，默认 `v5turbo`
- `inference_webui*.py` 的环境变量 `version` 默认 `v5turbo`
- `sample_steps` 对 v5turbo 默认 4 步；v4 仍为 32 步

v5Turbo 所需底模文件（请自行放置到这些路径）：

1. `GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5turbo.pth`
2. `GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth`

`s1v3.ckpt`、中文 RoBERTa、HuBERT 已存在可复用。

## 后端支持

`auto → cuda/ROCm → xpu → mps → dml → cpu`；无效设备名会警告后回退。可用环境变量 `device` 强制指定。

## 主补丁包含的文件

新增 3 个：`device_utils.py`、`module/cache_dit.py`、`module/models_v5.py`。

修改 15 个，核心包括：`TTS_infer_pack/TTS.py`（v5 版本识别、4 步采样、v5 mel 合成与声码器路径）、`module/models.py`（upstream CFM/DBCache）、`f5_tts/model/backbones/dit.py`（upstream DiT，支持 v5 无 step embedding）、`process_ckpt.py`（v5turbo 权重签名）、`configs/tts_infer.yaml`、`webui.py`、两个 `inference_webui*.py`、`config.py`，以及此前的通用后端修复文件。

共 18 个文件。`files/` 中是应用后应得到的完整文件；`MANIFEST.json` 有机器可读清单，`CHANGES.md` 有逐项说明。

## 未纳入补丁

`weight.json`（运行时音色选择，见 `optional/weight.json.patch`）、14 个角色权重、`user.dict`、生成音频、`runtime/` 下的环境变化与重编译 `.pyc`。

## 应用方法

```bash
cd /path/to/GPT-SoVITS-For-Intel
git -c core.autocrlf=false apply --check <path>/bsaireader-gpt-sovits.patch
git -c core.autocrlf=false apply    <path>/bsaireader-gpt-sovits.patch
```

脚本：`apply_patch.sh` / `apply_patch.ps1`。Windows 上请加 `core.autocrlf=false`。

## 验证

补丁从原版 ZIP 逐文件还原后应用成功，18 个输出文件与 `files/` 逐字节一致；全部 Python 文件通过编译；`TTS_Config` 能解析出 `version=v5turbo`、v5turbo 权重路径与设备 `xpu`；v5 模块导入正常。

尚未做真实音频合成验证，因为当前工作目录里缺少上述两个 v5turbo 底模文件。
