# GPT-SoVITS-For-Intel — BSAIReader 通用后端 + v5Turbo 补丁

由未修改的 `GPT-SoVITS-For-Intel.zip` 与已修改目录逐文件比对生成，包含后端通用化、v5Turbo 推理支持，以及此前的短句静音/缓存修复。

## v5Turbo 支持

- 默认版本 `v5turbo`，`custom.vits_weights_path` 指向 `gsv-v5-pretrained/s2Gv5turbo.pth`
- WebUI 版本选择提供 `v4` 与 `v5turbo`
- `TTS.py` 已移植 upstream 的 `module/models_v5.py`、`cache_dit.py`、DiT/CFM v5 路径
- `process_ckpt.py` 可识别 v5turbo 权重签名（`6bc5cb2a...`）
- `TTS_Config.save_configs()` 现在保留顶层 `version`，避免保存后静默回落到 v2（这是实测踩到的关键坑）

v5Turbo 需要：

```
GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5turbo.pth
GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth
```

## 后端支持

`auto → cuda/ROCm → xpu → mps → dml → cpu`；不可用设备会警告后回退。

## 主补丁文件

新增 3 个：`device_utils.py`、`module/cache_dit.py`、`module/models_v5.py`。修改 15 个，共 18 个文件。`files/` 是应用后的完整文件，`MANIFEST.json` 是机器可读清单，`CHANGES.md` 是逐项说明。

## 应用

```bash
cd /path/to/GPT-SoVITS-For-Intel
git -c core.autocrlf=false apply --check <path>/bsaireader-gpt-sovits.patch
git -c core.autocrlf=false apply    <path>/bsaireader-gpt-sovits.patch
```

## 验证

补丁可从原版 ZIP 还原后干净应用，18 个输出文件与 `files/` 逐字节一致；Python 编译通过；`TTS_Config` 解析为 `version=v5turbo`，保存再加载仍为 v5turbo。

## 本机速度参考（Intel Arc A770 / XPU / fp16）

- 模型加载：热缓存约 6.9 s，冷启动约 22.5 s
- 首次推理预热：约 90 s（kernel 编译）
- 22 字短句，8 步：约 12.85 s，音频 3.8 s，RTF 3.38（0.30× 实时）
- 22 字短句，4 步：约 11.09 s，音频 4.12 s，RTF 2.69（0.37× 实时）
- 142 字长文，8 步：约 28.7 s，音频 8.4 s，RTF 约 3.4

README 中 RX 9070 XT / ROCm 的 RTF 0.31 在这台 XPU 机器上无法复现；当前 runtime 的 IPEX 原生 DLL 加载失败，PyTorch XPU 回退路径明显更慢。
