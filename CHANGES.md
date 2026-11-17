# 修改明细

## 1. 默认版本固定为 v4

- `GPT_SoVITS/configs/tts_infer.yaml`
  - 新增顶层 `version: v4`。`TTS_Config` 读取的是顶层 `version`，原先缺失会回落到 `v2`；现在明确为 v4，默认配置也会取 `default_configs["v4"]`。
  - `custom.version: v2 → v4`。
  - 权重路径保持 `s1v3.ckpt` + `gsv-v4-pretrained/s2Gv4.pth`。
- `webui.py`
  - `version = "v4"`，忽略命令行传入的旧版本值（仅在无参数时补一个 v4 占位，避免语言参数解析变化）。
  - 版本单选框改为 `choices=["v4"], interactive=False`，界面上不再提供 v1/v2/v3 切换。
- `GPT_SoVITS/inference_webui.py`、`inference_webui_fast.py`
  - `os.environ.get("version", "v2") → "v4"`。

`api.py` 的旧版直连推理仍通过 `get_sovits_version_from_path_fast()` 从权重文件识别版本，这属于运行时探测，不需要写死。

## 2. 后端通用化

### 新增 `GPT_SoVITS/TTS_infer_pack/device_utils.py`

- `resolve_device(device)`：`auto` 或不可用设备按 `cuda → xpu → mps → dml → cpu` 回退，返回真正的 `torch.device`。
- `empty_device_cache(device)`：只在对应后端可用时调用其 `empty_cache()`。
- 所有可用性检查包在 `try/except` 中，兼容缺失 `torch.xpu` / `torch.backends.mps` / `torch_directml` 的构建。

### `GPT_SoVITS/TTS_infer_pack/TTS.py`

- 设备解析改用 `resolve_device()`，删除只处理 CUDA 不可用的旧分支。
- `set_device()` 接受字符串或 `torch.device`，先解析再保存，避免把 `"auto"` 写进配置。
- `set_ref_audio()` 清空缓存时补齐 `raw_sr` / `prompt_lang` / `aux_ref_audio_paths`，避免切换音色后辅助参考音频复用旧说话人。
- `empty_cache()` 委托给 `empty_device_cache()`。

### `GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py`

移除短句自动补 `。`/`.` 的逻辑——v4 权重下该前缀会导致 T2S 立即 EOS、输出近似静音。注释改为后端无关表述。

### `GPT_SoVITS/TTS_infer_pack/__init__.py`

改为惰性导入，避免仅导入 `device_utils` 就拉起整个模型栈和 IPEX。

### WebUI / API 入口

- `config.py`：`infer_device` 自动识别 cuda → xpu → mps → cpu。
- `api.py`：`-d` 帮助文本更新为 `cuda / xpu / mps / cpu`。
- `webui.py`：GPU 探测同时支持 CUDA/ROCm、XPU、MPS；IPEX 导入改为 `except Exception`。
- 两个 `inference_webui*.py`：设备由 `resolve_device(os.environ.get("device", "auto"))` 决定；XPU 种子与缓存调用加可用性保护。
- `t2s_lightning_module.py`：IPEX 导入从 `except ImportError` 改为 `except Exception`，兼容原生 DLL 加载失败。

## 3. 排除项

- `weight.json` 的改动只是运行时音色选择被清空，放在 `optional/weight.json.patch`。
- 6,758 个重编译 `.pyc`、335 个 runtime 新增文件、8 个删除文件（uvicorn 0.20.0→0.39.0、新增 emoji 2.15.0）、14 个角色权重、词典/生成音频均未纳入。详见 `MANIFEST.json`。
