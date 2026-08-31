from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from h3studio.backend_manager import BackendManager
from h3studio.errors import H3StudioError


class BackendManagerTests(unittest.TestCase):
    def test_resolves_manual_and_portable_layouts(self):
        with tempfile.TemporaryDirectory() as temp:
            portable = Path(temp)
            comfy = portable / "ComfyUI"
            comfy.mkdir()
            (comfy / "main.py").touch()
            _, resolved, python = BackendManager._resolve_layout(
                {"comfy_path": str(portable), "python_executable": sys.executable}
            )
            self.assertEqual(resolved, comfy)
            self.assertEqual(python, Path(sys.executable))

    def test_rejects_network_and_tls_overrides(self):
        with tempfile.TemporaryDirectory() as temp:
            comfy = Path(temp)
            (comfy / "main.py").touch()
            base = {
                "comfy_path": str(comfy),
                "python_executable": sys.executable,
                "port": 8189,
            }
            for extra in ("--listen 0.0.0.0", "--port=9000", "--tls-keyfile key.pem"):
                with self.assertRaises(H3StudioError):
                    BackendManager()._build_command({**base, "extra_args": extra})

    def test_managed_command_is_loopback_only(self):
        with tempfile.TemporaryDirectory() as temp:
            comfy = Path(temp)
            (comfy / "main.py").touch()
            command, cwd = BackendManager()._build_command(
                {
                    "comfy_path": str(comfy),
                    "python_executable": sys.executable,
                    "port": 8189,
                    "extra_args": "--preview-method auto",
                }
            )
            self.assertEqual(cwd, comfy)
            self.assertIn("127.0.0.1", command)
            self.assertIn("--disable-auto-launch", command)


if __name__ == "__main__":
    unittest.main()
