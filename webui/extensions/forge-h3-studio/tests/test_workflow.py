from __future__ import annotations

import unittest

from h3studio.errors import WorkflowValidationError
from h3studio.workflow import align_h3_frames, build_h3_workflow, normalize_request


def base_request(**changes):
    request = {
        "mode": "t2v",
        "model": "minimax_h3_fl2va.safetensors",
        "text_encoder": "qwen3vl_32b_minimax_h3.safetensors",
        "video_vae": "minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
        "prompt": "A slow dolly shot through a rainy neon street, distant traffic audio.",
        "width": 1344,
        "height": 768,
        "aspect_ratio": "16:9",
        "megapixels": 0.98,
        "rounding_multiple": 32,
        "frames": 124,
        "steps": 30,
        "seed": 123,
        "sampler": "euler",
        "scheduler": "simple",
        "shift_video": 12,
        "shift_audio": 3,
        "denoise": 1,
        "loras": [],
        "output": {"format": "auto", "codec": "auto", "fps": 24},
    }
    request.update(changes)
    return request


def nodes_of_type(workflow, class_type):
    return [(node_id, node) for node_id, node in workflow.items() if node["class_type"] == class_type]


class WorkflowTests(unittest.TestCase):
    def test_frame_alignment(self):
        self.assertEqual(align_h3_frames(5), 5)
        self.assertEqual(align_h3_frames(124), 124)
        self.assertEqual(align_h3_frames(125), 141)

    def test_t2v_builds_official_audio_video_chain(self):
        workflow, summary = build_h3_workflow(base_request())
        self.assertEqual(summary["mode"], "t2v")
        self.assertEqual(len(nodes_of_type(workflow, "MiniMaxH3ImageToVideo")), 1)
        self.assertEqual(len(nodes_of_type(workflow, "MiniMaxH3SigmaShift")), 1)
        self.assertEqual(len(nodes_of_type(workflow, "VAEDecode")), 1)
        self.assertEqual(len(nodes_of_type(workflow, "VAEDecodeAudio")), 1)
        self.assertEqual(len(nodes_of_type(workflow, "ImageFromBatch")), 1)
        self.assertEqual(len(nodes_of_type(workflow, "PreviewImage")), 1)
        self.assertEqual(len(nodes_of_type(workflow, "CreateVideo")), 1)
        save = nodes_of_type(workflow, "SaveVideo")[0][1]["inputs"]
        self.assertEqual(save["codec"], "auto")
        self.assertTrue(save["filename_prefix"].endswith("_seed_123"))
        self.assertEqual(summary["seed"], 123)
        self.assertEqual(summary["reproducible"]["seed"], 123)
        self.assertNotIn("_seed_123", summary["reproducible"]["output"]["filename_prefix"])
        self.assertNotIn("codec.encoding", save)

    def test_h264_uses_flat_v3_dynamic_combo_paths(self):
        workflow, _ = build_h3_workflow(
            base_request(output={"format": "mp4", "codec": "h264", "crf": 19, "fps": 24})
        )
        save = nodes_of_type(workflow, "SaveVideo")[0][1]["inputs"]
        self.assertEqual(save["codec"], "h264")
        self.assertEqual(save["codec.encoding"], "re-encode")
        self.assertEqual(save["codec.encoding.crf"], 19)

    def test_first_last_frame_mode(self):
        workflow, _ = build_h3_workflow(
            base_request(mode="fl2v", first_frame="forge_h3_studio/first.png", last_frame="forge_h3_studio/last.png")
        )
        conditioning = nodes_of_type(workflow, "MiniMaxH3ImageToVideo")[0][1]["inputs"]
        self.assertIn("first_frame", conditioning)
        self.assertIn("last_frame", conditioning)
        self.assertEqual(len(nodes_of_type(workflow, "LoadImage")), 2)

    def test_frame_crop_is_ratio_locked_and_scaled_before_h3(self):
        workflow, summary = build_h3_workflow(base_request(
            mode="i2v",
            first_frame="forge_h3_studio/first.png",
            first_frame_crop={
                "x": 100, "y": 80, "width": 800, "height": 600,
                "source_width": 1200, "source_height": 900,
            },
        ))
        crop = nodes_of_type(workflow, "ImageCrop")[0][1]["inputs"]
        scale = nodes_of_type(workflow, "ImageScale")[0]
        conditioning = nodes_of_type(workflow, "MiniMaxH3ImageToVideo")[0][1]["inputs"]
        self.assertAlmostEqual(crop["width"] / crop["height"], 1344 / 768, places=2)
        self.assertEqual(scale[1]["inputs"]["width"], 1344)
        self.assertEqual(scale[1]["inputs"]["height"], 768)
        self.assertEqual(conditioning["first_frame"], [scale[0], 0])
        self.assertGreater(summary["node_count"], 10)
        self.assertEqual(summary["reproducible"]["first_frame_crop"]["source_width"], 1200)

    def test_crop_outside_source_is_rejected(self):
        with self.assertRaises(WorkflowValidationError):
            build_h3_workflow(base_request(
                mode="i2v",
                first_frame="forge_h3_studio/first.png",
                first_frame_crop={
                    "x": 500, "y": 0, "width": 800, "height": 600,
                    "source_width": 1200, "source_height": 900,
                },
            ))

    def test_irrelevant_saved_slots_do_not_change_mode(self):
        workflow, _ = build_h3_workflow(
            base_request(first_frame="forge_h3_studio/old-first.png", last_frame="forge_h3_studio/old-last.png")
        )
        conditioning = nodes_of_type(workflow, "MiniMaxH3ImageToVideo")[0][1]["inputs"]
        self.assertNotIn("first_frame", conditioning)
        self.assertNotIn("last_frame", conditioning)
        self.assertEqual(len(nodes_of_type(workflow, "LoadImage")), 0)

    def test_reference_inputs_use_comfy_v3_autogrow_paths(self):
        refs = [
            {"kind": "image", "file": "forge_h3_studio/person.png"},
            {"kind": "video", "file": "forge_h3_studio/motion.mp4", "include_audio": True},
            {"kind": "audio", "file": "forge_h3_studio/voice.wav"},
        ]
        workflow, summary = build_h3_workflow(base_request(mode="ref", references=refs))
        conditioning = nodes_of_type(workflow, "MiniMaxH3ReferenceToVideo")[0][1]["inputs"]
        self.assertIn("ref_images.ref_image_0", conditioning)
        self.assertIn("ref_videos.ref_video_0", conditioning)
        self.assertIn("ref_video_audios.ref_video_audio_0", conditioning)
        self.assertIn("ref_audios.ref_audio_0", conditioning)
        self.assertEqual(len(nodes_of_type(workflow, "GetVideoComponents")), 1)
        self.assertTrue(any("<Picture 1>" in warning for warning in summary["warnings"]))

    def test_lora_order_and_clip_application(self):
        loras = [
            {"name": "h3_style.safetensors", "model_strength": 0.8, "enabled": True},
            {
                "name": "h3_character.safetensors",
                "model_strength": 1.1,
                "clip_strength": 0.35,
                "apply_to_clip": True,
                "enabled": True,
            },
        ]
        workflow, summary = build_h3_workflow(base_request(loras=loras))
        model_only = nodes_of_type(workflow, "LoraLoaderModelOnly")
        model_clip = nodes_of_type(workflow, "LoraLoader")
        self.assertEqual(len(model_only), 1)
        self.assertEqual(len(model_clip), 1)
        self.assertEqual(model_clip[0][1]["inputs"]["model"], [model_only[0][0], 0])
        conditioning = nodes_of_type(workflow, "MiniMaxH3ImageToVideo")[0][1]["inputs"]
        self.assertEqual(conditioning["clip"], [model_clip[0][0], 1])
        self.assertEqual(len(summary["loras"]), 2)

    def test_validation_and_warning_boundaries(self):
        with self.assertRaises(WorkflowValidationError):
            build_h3_workflow(base_request(mode="i2v"))
        with self.assertRaises(WorkflowValidationError):
            build_h3_workflow(base_request(mode="ref", references=[]))
        with self.assertRaises(WorkflowValidationError):
            build_h3_workflow(base_request(first_frame="../escape.png"))

        normalized = normalize_request(base_request(width=1350, height=777, frames=125))
        self.assertEqual(normalized["width"] % 32, 0)
        self.assertEqual(normalized["height"] % 32, 0)
        self.assertEqual(normalized["frames"], 141)

        custom = normalize_request(base_request(width=1350, height=777, rounding_multiple=16))
        self.assertEqual(custom["width"] % 16, 0)
        self.assertEqual(custom["height"] % 16, 0)
        self.assertEqual(custom["rounding_multiple"], 16)


if __name__ == "__main__":
    unittest.main()
