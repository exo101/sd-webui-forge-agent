# Forge MiniMax H3 Studio

一个面向 **Forge Neo** 的 MiniMax H3 本地视频工作台。扩展在 Forge 顶部增加独立的“MiniMax H3 工作台”页签；进入页签时可以自动启动本地 ComfyUI 后端并完成连接，不需要用户来回切窗口或手工拼节点。

> 当前版本：`0.2.2`。扩展本身不包含任何模型权重。

## 已实现功能

- Forge 顶级页签，三栏视频工作台界面，支持窄屏重排。
- 两种后端方式：
  - **Forge 托管**：进入 H3 页签时自动启动指定的 ComfyUI。
  - **外接 ComfyUI**：连接已经运行的本地或远程实例。
- ComfyUI WebSocket 真实运行状态：后端在提交前先建立事件通道，并把工作流总进度、当前节点/采样步、实时预览、队列位置、耗时和错误同步给前端；浏览器直连失败或刷新任务页也不会丢失状态。
- 任务中心：运行/完成筛选、排序、输出缩略图，以及模型、LoRA、Seed、分辨率、帧数、文件名和错误详情；输出图额外为 ComfyUI 自身任务历史提供首帧封面兜底。
- H3 生成模式：
  - 文生带音频视频（T2V）
  - 首帧图生视频（I2V）
  - 首尾帧视频（FL2V）
  - 多模态参考（REF：最多 9 图、3 视频、3 音频）
- 模型组件选择：H3 扩散模型、MiniMax 文本编码器、视频 VAE、音频 VAE。
- 参数：原生宽高比（1:1、2:3、3:2、3:4、4:3、9:16、16:9、21:9）、0.2–2.0 MP H3 尺寸表、可自由输入的取整倍数、手动宽高、帧数/秒数、Steps、Seed、Sampler、Scheduler、Video/Audio Shift、Denoise、参考图尺寸、导出格式、编码、CRF、FPS、位深。
- 帧数提交前自动对齐 H3 的 `17k+5` 网格，推荐范围提示为 124–362 帧，生成基准为 24 FPS。
- H3 LoRA 管理：后端目录扫描、搜索、添加、启停、排序、模型权重、可选文本编码器权重，以及 LoRA 组合预设。
- 素材库：图片、视频、音频上传；Windows 文件可直接拖到首帧、尾帧或对应参考槽位；生成结束后输出会自动归档到“已生成”，点击即可在大尺寸播放器中浏览或下载。
- 首尾帧联动取景：在原图上移动和缩放裁剪框，移动/缩放及关闭窗口都会保存位置，控制滑杆始终保留在窗口底部；框的比例始终跟随最终输出宽高，工作流通过 `ImageCrop` + `ImageScale` 无拉伸地送入 H3。
- Seed 可复现：实际 Seed 会显示在结果与素材属性中、写入视频元数据并加入输出文件名；任务可下载完整参数 JSON，也可一键把 Seed、模型、LoRA、裁剪和采样参数载回编辑器。
- 多参考提示词锚点：自动插入 `<Picture i>`、`<Video k>`、`<Audio j>`；每个参考视频可独立启用/忽略其音轨，音频序号会按 H3 的真实呈现顺序计算。
- 项目自动保存到浏览器本地存储，并可导入/导出项目 JSON；可导出实际提交给 ComfyUI 的 API 工作流 JSON。

## 运行结构

```text
Forge Neo
└─ MiniMax H3 工作台（本扩展）
   ├─ 前端：素材、参数、LoRA、任务、结果
   ├─ Forge 内置 FastAPI 桥接层
   └─ ComfyUI 推理后端
      ├─ H3 模型 / 文本编码器 / 视频 VAE / 音频 VAE
      ├─ H3 LoRA
      └─ ComfyUI output/video/Forge_H3_Studio_*.mp4
```

扩展不会在 Forge 进程内重复加载 H3 的 PyTorch 模型。Forge 负责界面与后端生命周期，ComfyUI 负责真正的模型加载和采样。托管后端只监听 `127.0.0.1`，扩展也会阻止额外参数覆盖监听地址、端口或 TLS 参数。由本扩展启动的后端会在 Forge 正常退出时尽力一并停止。

## 安装

1. 把整个 `forge-h3-studio` 文件夹放到：

   ```text
   stable-diffusion-webui-forge/extensions/forge-h3-studio
   ```

2. 准备一套包含 MiniMax H3 官方内置节点的较新 ComfyUI。启动后，`/object_info` 中至少应包含：

   ```text
   MiniMaxH3ImageToVideo
   MiniMaxH3ReferenceToVideo
   MiniMaxH3SigmaShift
   VAEDecodeAudio
   CreateVideo
   SaveVideo
   ```

3. 按 ComfyUI 标准目录放置兼容权重：

   ```text
   ComfyUI/models/diffusion_models/   H3 FL2VA / Ref2VA 扩散模型
   ComfyUI/models/text_encoders/      H3 MiniMax Qwen3-VL 文本编码器
   ComfyUI/models/vae/                H3 video VAE 与 audio VAE
   ComfyUI/models/loras/              H3 LoRA
   ```

   常见文件名会包含 `minimax_h3_fl2va`、`minimax_h3_ref2va`、`qwen3vl_32b_minimax_h3`、`minimax_h3_video_vae`、`minimax_h3_audio_vae`。量化版本的实际文件名可能不同，工作台会读取 ComfyUI 当前真实目录并让你选择，不依赖固定文件名。

4. 重启 Forge。扩展会自动补齐用于转发 ComfyUI 实时事件的轻量依赖；第一次进入“MiniMax H3 工作台”时会出现后端设置：

   - 扩展会先尝试发现 Forge 相邻目录、`ComfyUI`/`ComfyUI_windows_portable` 和 `COMFYUI_PATH` 环境变量。
   - Windows Portable：选择包含 `ComfyUI` 和 `python_embeded` 的根目录。
   - 手动安装：选择其中含 `main.py` 的 ComfyUI 目录；需要时手动指定 venv 的 Python。
   - 已经自己启动 ComfyUI：切换到“外接”并填写地址。

5. 保存设置，保持“进入 H3 页签时自动启动”开启。以后切换到工作台时，扩展会检查连接；尚未运行则启动后端，已经运行则直接复用。

## 基本使用流程

1. 进入 H3 工作台，等待右上角显示“`H3 已连接`”。
2. 选择 T2V、I2V、FL2V 或 REF 模式。
3. 图生/参考模式先导入素材，再点击或拖放到画布槽位。
4. 在右侧选择扩散模型、文本编码器、两个 VAE；开启自动匹配时，REF 优先匹配 Ref2VA，其他模式优先匹配 FL2VA。
5. 在 LoRA 页打开管理器，按从上到下的顺序建立加载栈并设置权重。大多数 H3 LoRA 只需模型权重；只有 LoRA 明确包含文本编码器权重时才开启“作用于编码器”。
6. 选择原生比例、百万像素和取整倍数。三项默认联动；也可关闭联动并手动输入宽高，提交尺寸仍按你指定的倍数取整。
7. I2V/FL2V 可把 Windows 图片直接拖到首尾帧卡片，再点“框选生成范围”；拖动取景框并用底部滑杆控制范围大小，直接关闭窗口也会保存。
8. 填写提示词和参数，点击“生成视频”。左侧任务页或右上角状态按钮会显示真实节点、采样进度和预览；完成后输出自动进入素材库，可点开全屏浏览、下载参数 JSON 或复用同一 Seed。

## 工作流实现说明

工作台生成的是 ComfyUI API prompt，主链路使用官方节点：

```text
UNETLoader ─ LoRA stack ─ MiniMaxH3SigmaShift ─ BasicGuider ─┐
CLIPLoader ─ LoRA stack ─ H3 Conditioning ──────────────────┤
RandomNoise + KSamplerSelect + BasicScheduler ───────────────┤
                                      SamplerCustomAdvanced ─┤
Video VAE ─ VAEDecode ────────────────────────────────────────┤
Audio VAE ─ VAEDecodeAudio ─ CreateVideo ─ SaveVideo ────────┘
```

H3 标准 BasicGuider 链路没有负面提示词和 CFG，因此界面没有放置无效的负面提示词/CFG 控件。H.264 的 `SaveVideo` v3 动态参数和多参考 Autogrow 参数均使用 ComfyUI API 所要求的点路径格式。

首尾帧存在取景设置时，输入链路为 `LoadImage → ImageCrop → ImageScale → MiniMaxH3ImageToVideo`。裁剪坐标使用原图像素，缩放节点只处理已经与输出同宽高比的区域，因此不会把人物横向或纵向压扁。

`examples/h3_request.json` 提供了一份便于二次开发或 API 调试的请求示例。

## RUM / FLUX.2 Klein 说明

本版本先完成 H3 的可运行闭环。工程已经把前端状态、模型目录读取、素材上传、任务队列和工作流构建分层，后续可以在同一工作台加入 **RUM-FLUX.2-klein-4B** 图片生成/多参考编辑模块，并把生成的关键帧直接送入 H3。RUM 使用的双文本投影和 teacher CLIP 链路不能当作普通 FLUX.2 Klein 工作流替换，因此不在 H3 图中“假装兼容”；需要单独的 RUM workflow adapter 后再接入。

## 当前边界

- 当前时间线是单镜头状态与素材概览，不是多段非线性剪辑器。
- 不负责下载模型，也不会绕过模型仓库的许可或访问要求。
- H3 节点属于较新的 ComfyUI 功能；若右上角已连接但提示缺少节点，请更新 ComfyUI，而不是安装同名的非官方 API 节点。
- 还未在本交付环境中进行真实 H3 权重推理（模型体积与 GPU 条件不具备）；工作流构建、后端桥接和安全边界已经由自动测试覆盖。

## 开发测试

在扩展目录执行：

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. python -m unittest discover -v -s tests
node --check javascript/h3_studio.js
```

## 许可

本扩展源码采用 MIT License。ComfyUI、Forge、MiniMax H3、RUM、FLUX.2、模型权重和 LoRA 各自遵循其原始许可。本实现为独立编写，用户提供的插件包仅用于确认交互需求，没有复制其源码或视觉资源。
