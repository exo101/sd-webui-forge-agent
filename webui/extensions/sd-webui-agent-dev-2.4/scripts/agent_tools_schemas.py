# =============================================================================
# Agent Tools — Function Calling 工具 schema 定义 (TOOLS, OpenAI 格式)
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "txt2img",
            "description": "根据文字描述生成图片。当用户要求画一张图、生成图片、创建图像时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "图片描述提示词，建议用英文，越详细越好"},
                    "negative_prompt": {"type": "string", "description": "负向提示词，不想出现的内容"},
                    "steps": {"type": "integer", "description": "采样步数，默认20"},
                    "width": {"type": "integer", "description": "图片宽度，默认1024"},
                    "height": {"type": "integer", "description": "图片高度，默认1024"},
                    "sampler_name": {"type": "string", "description": "采样器名称"},
                    "cfg_scale": {"type": "number", "description": "CFG引导强度，默认7.0"},
                    "seed": {"type": "integer", "description": "随机种子，-1为随机"},
                    "batch_size": {"type": "integer", "description": "每批生成数量，默认1"},
                    "n_iter": {"type": "integer", "description": "生成批次数量，默认1"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "img2img",
            "description": "基于参考图片和文字描述生成新图片。当用户上传了参考图片并要求修改/变换时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "图片描述提示词"},
                    "negative_prompt": {"type": "string", "description": "负向提示词"},
                    "denoising_strength": {"type": "number", "description": "重绘强度0-1，默认0.75"},
                    "steps": {"type": "integer", "description": "采样步数"},
                    "width": {"type": "integer", "description": "图片宽度。仅在用户明确指定尺寸时才填，否则留空（系统自动用参考图原始尺寸）"},
                    "height": {"type": "integer", "description": "图片高度。仅在用户明确指定尺寸时才填，否则留空（系统自动用参考图原始尺寸）"},
                    "sampler_name": {"type": "string", "description": "采样器名称"},
                    "cfg_scale": {"type": "number", "description": "CFG引导强度"},
                    "seed": {"type": "integer", "description": "随机种子"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_model",
            "description": "切换 Stable Diffusion 模型 (checkpoint)。当用户要求换模型、切换大模型时使用。可先用 list_models 查看可用模型。",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "模型文件名或名称关键词"}
                },
                "required": ["model_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_settings",
            "description": "修改 WebUI 的生图设置 (步数、CFG、采样器、尺寸、批量大小)。所有参数可选。",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {"type": "integer", "description": "采样步数"},
                    "cfg_scale": {"type": "number", "description": "CFG引导强度"},
                    "sampler_name": {"type": "string", "description": "采样器名称"},
                    "width": {"type": "integer", "description": "图片宽度"},
                    "height": {"type": "integer", "description": "图片高度"},
                    "batch_size": {"type": "integer", "description": "批量大小"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upscale",
            "description": "图片放大或调整大小。当用户要求放大图片、提高分辨率、调整图片尺寸、改大小、resize 时都使用此工具。需要用户上传图片或刚生成的图片。用户说调整大小/改尺寸时，用 resize_w 和 resize_h 指定目标宽高，不要用 scale。",
            "parameters": {
                "type": "object",
                "properties": {
                    "upscaler_name": {"type": "string", "description": "放大模型名称，如 4x-UltraSharp。未指定时默认优先使用 4x-UltraSharp，不要传 None"},
                    "scale": {"type": "number", "description": "放大倍数，默认2。仅当用户说放大几倍时使用"},
                    "resize_w": {"type": "integer", "description": "指定目标宽度像素。用户说调整大小/改尺寸/resize 到具体宽度时使用，覆盖 scale"},
                    "resize_h": {"type": "integer", "description": "指定目标高度像素。用户说调整大小/改尺寸/resize 到具体高度时使用，覆盖 scale"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_with_lora",
            "description": "使用 LoRA 模型生成图片。当用户要求使用某个 LoRA 风格/角色生成图片时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "基础提示词"},
                    "lora_name": {"type": "string", "description": "LoRA 名称 (不含扩展名)"},
                    "lora_weight": {"type": "number", "description": "LoRA 权重 0-2，默认0.8"},
                    "negative_prompt": {"type": "string", "description": "负向提示词"},
                    "steps": {"type": "integer", "description": "采样步数"},
                    "width": {"type": "integer", "description": "图片宽度"},
                    "height": {"type": "integer", "description": "图片高度"},
                    "seed": {"type": "integer", "description": "随机种子"},
                },
                "required": ["prompt", "lora_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_models",
            "description": "列出所有可用的 Stable Diffusion 主模型 (checkpoints)，直接扫描文件系统获取准确文件名。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_vae",
            "description": "列出所有可用的 VAE 模型。切换模型前建议先查看有哪些 VAE 可搭配。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_text_encoders",
            "description": "列出所有可用的文本编码器 (Text Encoders)。Krea2/Flux 等模型需要搭配特定 TE。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_controlnet",
            "description": "列出所有可用的 ControlNet 模型和预处理器（openpose, lineart, zoedepth 等）。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_guide",
            "description": "获取模型搭配指南：根据主模型推荐搭配的文本编码器(TE)、VAE 和 LoRA。切换模型前务必调用此工具了解正确组合。",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "可选，指定模型名称如 'krea2' 或 'flux'，不填则返回所有指南"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_vae",
            "description": "设置 VAE 模型。通常在切换主模型后根据 get_model_guide 的推荐来设置。",
            "parameters": {
                "type": "object",
                "properties": {
                    "vae_name": {"type": "string", "description": "VAE 文件名，如 'flux2-vae.safetensors'"},
                },
                "required": ["vae_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_text_encoder",
            "description": "设置文本编码器 (TE)（运行时热切换，无需重启）。Krea2 需要 qwen3vl_4b，Flux 需要 qwen_3_8b。",
            "parameters": {
                "type": "object",
                "properties": {
                    "te_name": {"type": "string", "description": "文本编码器文件名，如 'qwen3vl_4b_fp8_scaled.safetensors'"},
                },
                "required": ["te_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_model_components",
            "description": "【推荐】一键设置模型的全部组件：主模型+文本编码器+VAE（运行时热切换，无需重启）。只给 model_name 会自动匹配推荐的 TE/VAE。这是切换模型的首选工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "主模型文件名，如 'krea2_turbo_int8_convrot.safetensors'，只给它会自动匹配 TE/VAE"},
                    "te_name": {"type": "string", "description": "可选，指定文本编码器文件名"},
                    "vae_name": {"type": "string", "description": "可选，指定 VAE 文件名"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_samplers",
            "description": "列出所有可用的采样器。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_upscalers",
            "description": "列出所有可用的放大模型。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_settings",
            "description": "获取当前 WebUI 的生图设置信息 (模型、采样器、步数、CFG、尺寸等)。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_loras",
            "description": "列出已安装的 LoRA 模型。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
        {
            "type": "function",
            "function": {
                "name": "trellis2_image_to_3d",
                "description": "使用已安装的 TRELLIS.2 本地扩展将上传图片生成 GLB 三维模型。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seed": {"type": "integer", "description": "随机种子，默认0"},
                        "resolution": {"type": "string", "description": "网格分辨率，默认1024", "enum": ["512", "1024", "1536"]},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "video_keyframe_extract",
            "description": "从视频中提取关键帧。当用户上传视频并要求提取关键帧、截取画面、获取视频帧时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_frames": {"type": "integer", "description": "提取帧数，默认5"},
                    "method": {"type": "string", "description": "提取方式: even(均匀采样), first(首帧), last(尾帧), middle(中间帧)", "enum": ["even", "first", "last", "middle"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "video_to_frames",
            "description": "从视频中按时间间隔提取帧（每N秒一帧）。适合从长视频中批量提取画面。",
            "parameters": {
                "type": "object",
                "properties": {
                    "interval_seconds": {"type": "number", "description": "提取间隔（秒），默认1"},
                    "max_frames": {"type": "integer", "description": "最大提取帧数，默认20"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "h3_video_generate",
            "description": "【MiniMax H3 视频生成】生成视频。当用户要求'生成视频'/'制作视频'/'动起来'/'T2V'/'文生视频'/'图生视频'时使用。基于 forge-h3-studio 插件，支持云端 API 模式（已配置）和本地 ComfyUI 模式。生成的视频会保存到 outputs/h3_video 目录并在对话中展示。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "视频内容描述提示词（中英文均可，越详细越好）"},
                    "duration": {"type": "integer", "description": "视频时长（秒），范围 4-15，默认 5", "default": 5},
                    "aspect_ratio": {"type": "string", "description": "视频宽高比，默认 16:9", "enum": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"], "default": "16:9"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dreamina_video_generate",
            "description": "【Dreamina SeaDance 视频生成】用户明确使用 @dreamina-seedance-2-5-hc 或 @dreamina-seedance-2-0-hc 时必须使用。走 Agent 独立视频生成 API 设置，不依赖 forge-h3-studio、MiniMax Key 或本地 H3 模型。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "视频内容描述提示词"},
                    "duration": {"type": "integer", "description": "视频时长（秒），范围 4-15，默认 5", "default": 5},
                    "aspect_ratio": {"type": "string", "description": "视频比例", "enum": ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"], "default": "16:9"},
                    "model": {"type": "string", "description": "Dreamina 模型 ID，如 dreamina-seedance-2-0-hc 或 dreamina-seedance-2-5-hc"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stitch_images",
            "description": "把多张图片拼成一张网格图。当用户要求拼图、拼接多张图片时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {"type": "integer", "description": "列数，默认2"},
                    "padding": {"type": "integer", "description": "图片间距像素，默认10"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_preprocessors",
            "description": "列出所有可用的 ControlNet/ControlLLLite 预处理器。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_adetailer",
            "description": "ADetailer 脸部修复，自动检测并优化人脸细节。需要先上传或生成图片。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "修复时的提示词（可选）"},
                    "model_name": {"type": "string", "description": "检测模型名称（可选）"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_background",
            "description": "智能抠图工具。用户明确要求去除背景、抠图或透明背景时使用。不要用于图层分离；图层分离必须使用 layer_separation 工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["auto"], "description": "智能抠图", "default": "auto"},
                    "bg_color": {"type": "string", "description": "背景颜色: transparent/white/black，默认 transparent", "default": "transparent"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "layer_separation",
            "description": "图层分离：使用 sd-webui-see-through-sam 的 see-through 图层分离功能，把图像拆成可编辑图层。用户说'图层分离'、'分层'、'分离成 PSD 图层'时必须使用此工具，不要使用 remove_background。",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_format": {"type": "string", "enum": ["psd", "images"], "description": "输出形式。psd=优先合成为 PSD；images=返回分离图层图片", "default": "psd"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "api_image_generate",
            "description": "外部/API 图像生成工具。当前图像生成模型选择远程 API 模型时，用它进行文生图，不需要用户输入 @模型标签。支持 negative_prompt 和 quality 参数（YoboxAI gpt-image-2 等模型可用）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "图像生成提示词"},
                    "model": {"type": "string", "description": "API 模型 ID，默认读取设置区当前图像生成模型"},
                    "size": {"type": "string", "description": "输出尺寸（像素），默认 1024x1024。常用值：1024x1024(1:1正方形)、1024x1792(9:16竖版)、1792x1024(16:9横版)、768x1024(3:4)、1024x768(4:3)。用户提到竖版/9:16/竖屏时必须传 1024x1792；横版/16:9/横屏时传 1792x1024", "default": "1024x1024"},
                    "negative_prompt": {"type": "string", "description": "负向提示词，描述不希望出现的元素，如 'blurry, low resolution, ugly, deformed, watermark'（可选）", "default": ""},
                    "quality": {"type": "string", "description": "生成质量档位：high/medium/low（可选，仅部分模型支持）", "default": ""},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "api_image_edit",
            "description": "外部/API 图像编辑工具。用户使用 @banana2、@bananapro、@gpt-image-2、@gpt-image-2.5、@Qwen-Image-Edit-2511、@FireRed-Image-Edit 等 API 图像编辑模型标签时必须优先使用此工具。不要改用 remove_background、change_background 或本地 Klein 编辑。",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "图像编辑指令，如 'change the image background to pure white, preserve the subject exactly'"},
                    "model": {"type": "string", "description": "API 模型 ID。若用户标签指定了模型，必须填写该模型 ID，如 banana2"},
                    "size": {"type": "string", "description": "输出尺寸（像素），默认 auto(保持原图尺寸)。如需指定比例：1024x1792(9:16竖版)、1792x1024(16:9横版)、1024x1024(1:1)", "default": "auto"},
                },
                "required": ["instruction", "model"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_image",
            "description": "通用图像编辑：根据文字指令编辑图片（换背景/改风格/加物体/去物体等）。会自动切换到 Flux.2-Klein 多模态编辑模型执行编辑，完成后自动切回原模型，用户无感。用户说'把这个改成...'/'加上...'/'去掉...'/'修改...'时使用。【重要】换背景：如果用户要的背景不在预设氛围选项（night/sunset/rainy/snowy/foggy/cyberpunk/morning/studio）中，必须使用此工具，不能使用 change_background。",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "编辑指令，如 'change background to night scene, preserve subject'、'add a cat on the table'"},
                    "strength": {"type": "number", "description": "编辑强度 0.0-1.0，默认 0.6，值越大改动越大", "default": 0.6},
                },
                "required": ["instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_background",
            "description": "换背景氛围专用工具：将图片的背景/氛围替换为预设风格（night/sunset/rainy/snowy/foggy/cyberpunk/morning/studio 之一）。自动使用 Klein 编辑模型，保留主体，只改背景。用户说'换成夜晚'/'换个背景'/'改成雨天'时优先使用。【注意】此工具只能使用预设氛围，如果用户想要的背景不在预设列表中（如卧室、海边、森林等），请使用 edit_image 工具，不要使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "atmosphere": {"type": "string", "enum": ["night", "sunset", "rainy", "snowy", "foggy", "cyberpunk", "morning", "studio"], "description": "目标氛围预设", "default": "night"},
                },
                "required": ["atmosphere"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_extensions",
            "description": "列出所有已安装的扩展插件及其状态。当用户询问有哪些插件时使用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "联网搜索实时新闻、最新资料和公开网页信息。用户问今天/最新/实时/新闻/网络资料/帮我查一下时优先使用，并在回答中给出来源链接。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，建议包含地点、主题和时间范围"},
                    "max_results": {"type": "integer", "description": "返回结果数量，默认6，最多10", "default": 6},
                    "region": {"type": "string", "description": "搜索区域，如 zh-cn、us-en，默认 zh-cn", "default": "zh-cn"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_read_url",
            "description": "读取指定网页内容并提取正文。适合用户给出链接，或 web_search 后需要核对来源细节时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "要读取的 http/https 网页地址"},
                    "max_chars": {"type": "integer", "description": "最多提取字符数，默认8000，最多20000", "default": 8000},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_extension",
            "description": "深入研究某个扩展插件的真实功能。读取扩展的 README.md 和脚本文件，了解它实际是干什么的。【重要】当你不确定某个扩展/功能/模型是做什么的时，必须先调用此工具调查，绝对不能编造！",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "扩展名称或关键词，如 'seedvr2', 'controlnet', 'adetailer', 'rembg'",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explore_webui",
            "description": "探索 WebUI 的目录结构和内置功能。当用户问到 WebUI 有什么功能、某个目录是干什么的时，调用此工具调查。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对路径，默认为 '.' 表示 Forge Neo 整合包根目录。如 'webui/extensions', 'webui/models', 'webui/modules'",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_workspace_file",
                    "description": "读取 Forge Neo 整合包根目录内的代码、配置和文本文件内容。用户要求查看文件、解释脚本、核对配置或读取 README 时使用；路径必须相对整合包根目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "整合包根目录内的相对文件路径，如 'webui/modules/shared.py'、'webui/extensions/插件名/README.md'"},
                    "max_chars": {"type": "integer", "description": "最多读取的字符数，默认 12000，最大 999999", "default": 12000, "maximum": 999999},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_document",
                    "description": "提取并分析 Forge Neo 整合包根目录内的文档内容。支持 TXT、MD、JSON、CSV、DOCX，以及环境具备解析组件时的 PDF。用户要求总结、审阅、解释或从文档中找信息时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "整合包根目录内的相对文档路径"},
                    "question": {"type": "string", "description": "用户希望从文档中回答的问题或分析角度", "default": ""},
                    "max_chars": {"type": "integer", "description": "最多提取的字符数，默认 16000，最大 999999", "default": 16000, "maximum": 999999},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "audit_extensions",
            "description": "汇总所有已安装插件的启用状态、关键脚本和 README 摘要。用户问“我的插件都能做什么”“帮我梳理所有插件”时优先使用；若需了解一个特定插件，再使用 research_extension。",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_readme": {"type": "boolean", "description": "是否包含每个插件的 README 摘要，默认 true", "default": True},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repair_workspace_file",
                    "description": "修复 Forge Neo 整合包根目录内的代码、配置或文本文件。必须先读取当前文件，再用 old_text 精确替换为 new_text；修改前自动创建可恢复备份。仅在用户授权修复、安装或调整整合包时使用，不修改密钥文件，不允许路径越界。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "整合包根目录内的相对文件路径"},
                    "old_text": {"type": "string", "description": "从当前文件读取到的、需要替换的完整原文，必须恰好匹配一处"},
                    "new_text": {"type": "string", "description": "替换后的完整文本"},
                    "reason": {"type": "string", "description": "本次修复原因"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diagnose_workspace",
                    "description": "对 Forge Neo 整合包根目录或指定 Python 文件执行无破坏诊断（Python 语法/编译检查），用于定位并验证报错。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "整合包根目录内的 Python 文件或目录，默认为 '.'"},
                },
            },
        },
    },
]
