"""Install dependencies for forge-h3-studio.

- websocket-client: ComfyUI 进度回传（managed/external 模式需要）
- diffsynth[audio,quant]: local 模式本地推理 MiniMax H3 的核心依赖
  （audio = 音频 VAE 支持，quant = NF4/INT8 量化推理支持）
"""

import launch


try:
    import websocket  # noqa: F401
except Exception:
    launch.run_pip(
        'install "websocket-client>=1.8,<2"',
        "Forge H3 Studio ComfyUI progress relay",
    )


# diffsynth 是 local 模式（Forge 进程内 DiffSynth 本地推理）的核心依赖。
# 即使当前用的是 api/managed/external 模式，提前装好也不会有副作用，
# 切换到 local 模式时即可直接使用。
try:
    import diffsynth  # noqa: F401
    from diffsynth.configs import MODEL_CONFIGS  # noqa: F401
    # 确认 MODEL_CONFIGS 非空（某些旧版 diffsynth 可能没有内置 H3 配置）
    if not MODEL_CONFIGS:
        raise ImportError("diffsynth MODEL_CONFIGS 为空，可能版本过旧")
except Exception:
    launch.run_pip(
        'install "diffsynth[audio,quant]" --upgrade',
        "Forge H3 Studio local inference engine (diffsynth)",
    )
