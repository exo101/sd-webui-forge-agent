"""Install the small optional dependency used for reliable ComfyUI event relay."""

import launch


try:
    import websocket  # noqa: F401
except Exception:
    launch.run_pip(
        'install "websocket-client>=1.8,<2"',
        "Forge H3 Studio ComfyUI progress relay",
    )
