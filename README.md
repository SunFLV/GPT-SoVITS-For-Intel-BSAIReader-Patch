# GPT-SoVITS-For-Intel — BSAIReader 通用后端 + v5Turbo 补丁

由未修改的 `GPT-SoVITS-For-Intel.zip` 与已修改目录逐文件比对生成，包含后端通用化、v5Turbo 推理支持、逐句批量推理接口，以及此前的短句静音/缓存修复。

## 动态接活 + 逐句流式（`POST /tts_batch_stream`）

长文的真正瓶颈不是"批大小"，而是"槽位空转"：AR 是所有序列同步走一步，一批里
只有最长那句决定耗时，先 EOS 的短句只能 padding 陪跑。`/tts_batch_stream` 把
空出来的槽位立刻补给下一句，并且**每完成一句就回传一行**：

```
{"index": 0, "wav": "<base64>"}
{"index": 3, "wav": "..."}          <- 完成顺序 != 句序，必须按 index 归位
```

实测 16 句长短混排：第 1 句 19.4 s 就能播（静态批要等整批 60 s+），全部 16 句
136.3 s 跑完，音频时长与句长相关性 0.94。

## 逐句批量推理（`POST /tts_batch`）

长文合成的瓶颈是逐句串行的自回归解码：`/tts` 的推理在事件循环里同步执行，请求之间严格串行，`batch_size` 只作用于“这一句被切成几段”。新增的 `/tts_batch` 接受 `texts: [...]`，把这几句凑成一次前向，再按句返回音频（`wavs` 与 `texts` 同序）—— 调用方仍拿到逐句 WAV，点读时间点不变。

| 一次提交 | 每句耗时（A770/v5turbo） | 相对逐句 |
|---|---|---|
| 1 句 | 12.1 s | 1.0× |
| 4 句 | 8.2 s | 1.5× |
| 8 句 | 5.6 s | 2.1× |
| 12/16 句 | 6.5 s | 1.9× |

## v5Turbo 支持

- 默认版本 `v5turbo`，`custom.vits_weights_path` 指向 `gsv-v5-pretrained/s2Gv5turbo.pth`
- WebUI 版本选择提供 `v4` 与 `v5turbo`
- `TTS.py` 已移植 upstream 的 `module/models_v5.py`、`cache_dit.py`、DiT/CFM v5 路径
- `process_ckpt.py` 可识别 v5turbo 权重签名（`6bc5cb2a...`）
- `TTS_Config.save_configs()` 现在保留顶层 `version`，避免保存后静默回落到 v2（这是实测踩到的关键坑）
- 权重改为 `mmap=True` + `map_location=xpu` 直接映射到设备，避免先在 CPU 内存反序列化出一份完整副本（SoVITS 权重实测省约 2 GiB 私有内存），mmap 不支持时自动回退到原加载方式

v5Turbo 需要：

```
GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5turbo.pth
GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth
```

## 后端支持

`auto → cuda/ROCm → xpu → mps → dml → cpu`；不可用设备会警告后回退。

## 主补丁文件

新增 4 个：`bsai_capabilities.py`、`device_utils.py`、`module/cache_dit.py`、`module/models_v5.py`。修改 16 个（含 `api_v2.py`），共 20 个文件。`files/` 是应用后的完整文件，`MANIFEST.json` 是机器可读清单，`CHANGES.md` 是逐项说明。

## 自报能力：`/api/voices`、`/api/models`、`/api/capabilities`

**这台机器支持哪些模型、有哪些音色，由后端自己回答**，不需要额外部署前端或代理层。

补丁在 `api_v2.py` 里挂载三条只读路由（`bsai_capabilities.py` 提供）：

| 路由 | 内容 |
|---|---|
| `GET /api/voices` | `{voices, versions, default_version, backend_capabilities, server_name, voice_details}` —— 与 bsai 代理层**完全同构**，前端读到任意一家的返回都长一样 |
| `GET /api/models` | 本机可用的模型版本 + 标签 |
| `GET /api/capabilities` | 完整清单（含补丁特性、音色明细、后端路径） |

数据来自两个纯数据文件（安装时生成，不改后端源码）：

```json
// BSAI_PATCH.json —— 补丁声明
{"patch": "...", "models": ["v4", "v5turbo"],
 "features": {"batch_synthesize": "/tts_batch", "capabilities": "/api/capabilities", ...},
 "server_name": "本机 A770"}

// BSAI_VOICES.json —— 音色清单（路径相对后端根目录，可跨机器）
{"voices": [{"name": "v4_default", "description": "高松灯-叙述", "text_lang": "zh",
             "prompt_lang": "ja", "ref_audio_path": "bsai/voices/default_ref.mp3",
             "prompt_text": "そよちゃん、部活でコントラバス、だっけ"}]}
```

生成时会和 `GPT_SoVITS/configs/tts_infer.yaml` 里真实存在的版本段取交集，
所以**只装了一半补丁的机器不会误报 v5turbo**。

音色来源按优先级：`--voices-json` 指定（bsai 的 `config_intel.yaml` 也能直接用，
YAML/JSON 都认）→ 已有 `BSAI_VOICES.json` → `bsai/config_intel.yaml` → 扫
`voices/ ref_audios/ SunFLV/Reference_Voices/`。绝对路径若在后端目录外，
会自动在根目录里找同名副本并改写成相对路径，换台机器也能用。

### 顺带：可以直接用音色名合成

`/tts`、`/tts_batch`、`/tts_batch_stream` 现在都接受 `voice`（或把音色名当
`ref_audio_path` 传）：

```json
{"text": "测试", "voice": "v4_tomori"}
```

后端按 `BSAI_VOICES.json` 解析出参考音路径与 `prompt_text / prompt_lang /
text_lang`，调用方不必知道那台机器的文件布局。传真实文件路径时行为不变。

### 安装/重建

```bash
python write_capabilities.py <后端根目录>
python write_capabilities.py <后端根目录> --voices-json D:/BSAIReader/server/config_intel.yaml --name "本机 A770"
python bsai_capabilities.py          # 命令行自查当前清单
```

`apply_patch.sh` / `apply_patch.ps1` 应用成功后会自动调用一次（名字默认取机器名）。
池服务（`optional/generic-pool-service`）也挂了同样三条路由，且 `/api/voices`
不需要 `X-API-Key`（只读元数据），方便前端在服务器下拉里直接读。

NVIDIA P106-90 3G 的 INT8 部署是独立可选附件，见
`optional/nvidia-int8/`：离线量化 `s2Gv5turbo_int8.pth`，用单独的
`runtime_cuda`（torch 2.5.1+cu121）在 9891 起单实例池，不动原 XPU 运行时。

## 应用

```bash
cd /path/to/GPT-SoVITS-For-Intel
git -c core.autocrlf=false apply --check <path>/bsaireader-gpt-sovits.patch
git -c core.autocrlf=false apply    <path>/bsaireader-gpt-sovits.patch
```

## 验证

补丁可从原版 ZIP 还原后干净应用，19 个输出文件与 `files/` 逐字节一致；Python 编译通过；`TTS_Config` 解析为 `version=v5turbo`，保存再加载仍为 v5turbo。

## 本机速度参考（Intel Arc A770 / XPU / fp16）

- 模型加载：热缓存约 6.9 s，冷启动约 22.5 s
- 首次推理预热：约 90 s（kernel 编译）
- 22 字短句，8 步：约 12.85 s，音频 3.8 s，RTF 3.38（0.30× 实时）
- 22 字短句，4 步：约 11.09 s，音频 4.12 s，RTF 2.69（0.37× 实时）
- 142 字长文，8 步：约 28.7 s，音频 8.4 s，RTF 约 3.4

README 中 RX 9070 XT / ROCm 的 RTF 0.31 在这台 XPU 机器上无法复现；当前 runtime 的 IPEX 原生 DLL 加载失败，PyTorch XPU 回退路径明显更慢。
