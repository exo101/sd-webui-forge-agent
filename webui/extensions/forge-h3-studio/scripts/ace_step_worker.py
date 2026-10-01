"""ACE-Step worker executed by breeze-tts/venv, outside the Forge process."""
import json
import os
import sys
import types
from pathlib import Path

os.environ["H3_ACE_STEP_WORKER"] = "1"
def main():
    # This file lives under Forge's scripts directory.  Forge imports every
    # .py file there, so imports and execution must be deferred until this
    # file is actually launched by the isolated worker interpreter.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    worker_script_dir = Path(__file__).resolve().parent
    webui_root = worker_script_dir.parent.parent.parent
    sys.path.insert(0, str(webui_root))
    sys.path.insert(0, str(worker_script_dir.parent / "ACE-Step-1.5"))

    # ace_step_ui only needs these output options from Forge.  Do not import
    # the full Forge runtime in the isolated worker.
    shared_stub = types.SimpleNamespace(
        opts=types.SimpleNamespace(
            outdir_samples="",
            outdir_txt2img_samples=str(webui_root / "outputs" / "txt2img-images"),
        )
    )
    modules_stub = types.ModuleType("modules")
    modules_stub.shared = shared_stub
    sys.modules["modules"] = modules_stub

    from ace_step_ui import generate_music
    import torch
    import transformers

    print(
        "[ACE-Step worker] Python=%s | torch=%s | CUDA=%s | transformers=%s"
        % (sys.executable, torch.__version__, torch.cuda.is_available(), transformers.__version__),
        flush=True,
    )
    try:
        import flash_attn
        print("[ACE-Step worker] flash_attn=loaded", flush=True)
    except Exception as exc:
        print(f"[ACE-Step worker] flash_attn=unavailable: {exc}", flush=True)

    request = json.loads(sys.argv[-1])
    path, error = generate_music(**request)
    print("H3_ACE_RESULT=" + json.dumps({"path": path, "error": error}, ensure_ascii=False))
    return 0 if path and not error else 1


if __name__ == "__main__":
    sys.exit(main())
