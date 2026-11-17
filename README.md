# GPT-SoVITS-For-Intel — BSAIReader 通用后端补丁（默认 v4）

补丁由未修改的 `GPT-SoVITS-For-Intel.zip` 与已修改目录逐文件比对生成，并完成两项收尾：**后端通用化** 与 **默认模型版本固定为 v4**。

## 默认模型版本：v4

已写死为 v4，不再默认或可选 v2：

- `GPT_SoVITS/configs/tts_infer.yaml`：新增顶层 `version: v4`，`custom.version` 也改为 `v4`。
- `webui.py`：`version = "v4"`，版本单选框只保留 `v4` 且不可切换。
- `GPT_SoVITS/inference_webui.py` / `inference_webui_fast.py`：环境变量 `version` 默认值改为 `v4`。
- `api.py`（旧版直连接口）仍按权重文件自动识别版本，这是运行时探测而非默认值，保持不变。

YAML 中 v1/v2/v3 的配置段仍保留在文件里，但不会被默认使用。

## 支持的后端

| 后端 | 支持情况 |
| --- | --- |
| CPU | 原生支持；`is_half` 视为 False |
| NVIDIA CUDA / AMD ROCm | 通过 `torch.cuda` 自动识别 |
| Intel XPU | 保留原有行为；`torch.xpu` 不可用时自动跳过 |
| Apple MPS | 通过 `torch.backends.mps` 自动识别 |
| DirectML | 安装 `torch-directml` 后可用 `dml` |

设备选择：`auto → cuda/rocm → xpu → mps → dml → cpu`。显式指定不可用设备会警告后回退，而不是崩溃。可用环境变量 `device` 强制指定，例如 `device=cpu python webui.py`。

## 主补丁包含的文件

**新增**
- `GPT_SoVITS/TTS_infer_pack/device_utils.py`：`resolve_device()` / `empty_device_cache()`

**核心推理**
- `TTS.py`：自动设备解析、切换音色时清空全部 prompt 缓存、按后端安全释放缓存
- `TextPreprocessor.py`：移除短句补标点（静音修复），注释改为后端无关
- `TTS_infer_pack/__init__.py`：惰性导入，避免只导入设备工具就拉起模型栈
- `configs/tts_infer.yaml`：`device: auto`、`version: v4`、v4 权重路径

**入口与启动器**
- `config.py`、`api.py`、`webui.py`
- `GPT_SoVITS/inference_webui.py`、`inference_webui_fast.py`
- `GPT_SoVITS/AR/models/t2s_lightning_module.py`

共 11 个文件（含 1 个新增）。逐文件说明见 `CHANGES.md`，机器可读清单见 `MANIFEST.json`。

## 未纳入补丁的内容

- `weight.json`：仅 WebUI 音色选择被清空，属于运行时状态；独立补丁在 `optional/weight.json.patch`。
- 14 个角色权重（约 1.1 GB）、`user.dict`、生成音频：用户数据，不应进 Git。
- `runtime/` 下 6,758 个重编译 `.pyc`、335 个新增文件、8 个删除文件：环境变化（uvicorn 0.20.0→0.39.0、新增 emoji 2.15.0），不是源码修改。

## 应用方法

在未修改的原版解压目录中执行：

```bash
cd /path/to/GPT-SoVITS-For-Intel
git -c core.autocrlf=false apply --check D:/BSAIReader/GPT-SoVITS-For-Intel-BSAIReader-Patch/bsaireader-gpt-sovits.patch
git -c core.autocrlf=false apply    D:/BSAIReader/GPT-SoVITS-For-Intel-BSAIReader-Patch/bsaireader-gpt-sovits.patch
```

也可使用 `apply_patch.sh` 或 `apply_patch.ps1`。`core.autocrlf=false` 很重要：Windows 默认配置会改写补丁行尾。`files/` 中保存了应用后应得到的完整文件，并保留各文件原有的行尾风格。

## 推送到 GitHub 建议

只提交这 11 个源码/配置文件，不要提交 `runtime/`、模型权重、`.pyc`、生成音频。提交信息建议：

```
Make GPT-SoVITS patch backend-agnostic and default to v4
```

## 验证记录

- 补丁应用到从原版 ZIP 逐文件还原的干净目录，11 个输出文件与 `files/` 逐字节一致。
- 全部 Python 文件通过 `py_compile`（仅有上游既存 `SyntaxWarning`）。
- 模拟 CPU-only / CUDA-only / MPS-only / XPU-only 环境测试设备解析与缓存释放，均正确回退。
- 真实 `torch 2.7.0+xpu` 上 `auto → xpu`；`TTS_Config` 加载配置得到 `version=v4`、设备 `xpu`、v4 权重路径。
