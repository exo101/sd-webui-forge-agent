from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from h3studio.comfy_client import ComfyClient, ComfyEventStream, normalize_base_url, safe_media_path
from h3studio.errors import H3StudioError


def node_info(option_name, values):
    return {"input": {"required": {option_name: [values, {}]}}}


class ClientTests(unittest.TestCase):
    def test_event_stream_parses_json_and_binary_previews(self):
        class Connection:
            def __init__(self):
                self.payloads = [
                    '{"type":"executing","data":{"node":"7"}}',
                    (1).to_bytes(4, "big") + (2).to_bytes(4, "big") + b"png-bytes",
                ]

            def recv(self):
                if self.payloads:
                    return self.payloads.pop(0)
                raise RuntimeError("closed")

            def close(self):
                pass

        events = []
        stream = ComfyEventStream(Connection(), events.append).start()
        stream._thread.join(timeout=2)
        self.assertEqual(events[0]["type"], "executing")
        self.assertEqual(events[1]["type"], "preview_image")
        self.assertEqual(events[1]["data"]["mime"], "image/png")
        self.assertEqual(events[1]["data"]["bytes"], b"png-bytes")

    @patch("h3studio.comfy_client.httpx.Client")
    def test_media_proxy_forwards_video_range(self, client_class):
        transport = client_class.return_value
        response = MagicMock()
        response.is_success = True
        response.status_code = 206
        response.headers = {"content-type": "video/mp4", "content-range": "bytes 0-99/1000"}
        response.iter_bytes.return_value = iter([b"video"])
        transport.send.return_value = response
        client = ComfyClient(base_url="http://127.0.0.1:8189")

        opened, iterator = client.open_media("clip.mp4", "video", "output", "bytes=0-99")

        self.assertIs(opened, response)
        _, kwargs = transport.build_request.call_args
        self.assertEqual(kwargs["headers"], {"Range": "bytes=0-99"})
        self.assertEqual(list(iterator), [b"video"])

    def test_url_and_media_path_validation(self):
        self.assertEqual(normalize_base_url("http://127.0.0.1:8189/"), "http://127.0.0.1:8189")
        self.assertEqual(safe_media_path("clip.mp4", "video/run-1"), ("clip.mp4", "video/run-1"))
        for value in ("file:///tmp/comfy", "127.0.0.1:8189", "http://user:pass@localhost:8189"):
            with self.assertRaises(H3StudioError):
                normalize_base_url(value)
        with self.assertRaises(H3StudioError):
            safe_media_path("../clip.mp4", "")
        with self.assertRaises(H3StudioError):
            safe_media_path("clip.mp4", "../outside")

    def test_catalog_extracts_choices_and_reports_missing_h3_nodes(self):
        client = ComfyClient(base_url="http://127.0.0.1:8189")
        info = {
            "UNETLoader": node_info("unet_name", ["h3_fl2va.safetensors"]),
            "CLIPLoader": node_info("clip_name", ["h3_clip.safetensors"]),
            "VAELoader": node_info("vae_name", ["video_vae.safetensors", "audio_vae.safetensors"]),
            "LoraLoader": node_info("lora_name", ["h3_turbo.safetensors"]),
            "KSamplerSelect": node_info("sampler_name", ["euler"]),
            "BasicScheduler": node_info("scheduler", ["simple"]),
        }
        client.object_info = lambda: info
        catalog = client.catalog()
        self.assertEqual(catalog["models"], ["h3_fl2va.safetensors"])
        self.assertEqual(catalog["loras"], ["h3_turbo.safetensors"])
        self.assertFalse(catalog["h3_ready"])
        self.assertIn("MiniMaxH3ImageToVideo", catalog["missing_nodes"])


if __name__ == "__main__":
    unittest.main()
