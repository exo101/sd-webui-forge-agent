import os
import base64
import html
from io import BytesIO

import av
import gradio as gr
import numpy as np
from PIL import Image
from tqdm import tqdm

from modules import devices, images, infotext_utils, scripts, scripts_postprocessing, shared, ui_common, ui_tempdir
from modules.shared import opts


def _image_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_before_after_comparison(before: Image.Image | None, after: Image.Image | None) -> str:
    if before is None or after is None:
        return ""

    before = before.convert("RGBA")
    after = after.convert("RGBA")
    if before.size != after.size:
        after = after.resize(before.size, Image.Resampling.LANCZOS)

    before_src = _image_data_uri(before)
    after_src = _image_data_uri(after)
    width, height = before.size
    return f"""
<div class="extras-before-after" data-split="50">
  <div class="extras-before-after-stage" style="aspect-ratio: {width} / {height};">
    <img class="extras-before-after-img" src="{before_src}" alt="原图">
    <div class="extras-before-after-overlay" style="width: 50%;">
      <img class="extras-before-after-img" src="{after_src}" alt="生成结果">
    </div>
    <div class="extras-before-after-bar" style="left: 50%;"><span></span></div>
    <input class="extras-before-after-range" type="range" min="0" max="100" value="50"
      aria-label="拖动查看原图和生成结果对比"
      oninput="const p=this.value+'%'; const root=this.closest('.extras-before-after'); root.querySelector('.extras-before-after-overlay').style.width=p; root.querySelector('.extras-before-after-bar').style.left=p;">
  </div>
</div>
"""


def _file_url(path: os.PathLike | str | None) -> str:
    if not path:
        return ""
    path = os.path.abspath(os.fspath(path))
    if shared.demo is not None and os.path.isfile(path):
        ui_tempdir.register_tmp_file(shared.demo, path)
    return f"/gradio_api/file={path.replace(chr(92), '/')}"


def _render_video_before_after_comparison(before_path: os.PathLike | str | None, after_path: os.PathLike | str | None) -> str:
    before_src = html.escape(_file_url(before_path), quote=True)
    after_src = html.escape(_file_url(after_path), quote=True)
    if not before_src or not after_src:
        return ""
    sync_js = (
        "const root=this.closest('.extras-before-after');"
        "const top=root.querySelector('.extras-before-after-overlay video');"
        "if(top&&Math.abs((top.currentTime||0)-(this.currentTime||0))>0.12)top.currentTime=this.currentTime;"
    )
    return f"""
<div class="extras-before-after extras-before-after-video" data-split="50">
  <div class="extras-before-after-stage">
    <video class="extras-before-after-img" src="{before_src}" controls muted playsinline preload="metadata"
      onplay="{sync_js} const top=this.closest('.extras-before-after').querySelector('.extras-before-after-overlay video'); if(top)top.play();"
      onpause="const top=this.closest('.extras-before-after').querySelector('.extras-before-after-overlay video'); if(top)top.pause();"
      onseeking="{sync_js}"
      ontimeupdate="{sync_js}"></video>
    <div class="extras-before-after-overlay" style="width: 50%;">
      <video class="extras-before-after-img" src="{after_src}" muted playsinline preload="metadata"></video>
    </div>
    <div class="extras-before-after-bar" style="left: 50%;"><span></span></div>
    <input class="extras-before-after-range" type="range" min="0" max="100" value="50"
      aria-label="拖动查看原视频和生成结果对比"
      oninput="const p=this.value+'%'; const root=this.closest('.extras-before-after'); root.querySelector('.extras-before-after-overlay').style.width=p; root.querySelector('.extras-before-after-bar').style.left=p;">
  </div>
</div>
"""


def _normalize_video_input(video_input):
    if isinstance(video_input, (tuple, list)):
        video_input = video_input[0] if video_input else None
    if isinstance(video_input, dict):
        video_input = video_input.get("name") or video_input.get("path")
    if hasattr(video_input, "name"):
        video_input = video_input.name
    return os.fspath(video_input) if video_input else None


def run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, _, *args, save_output: bool = True):
    devices.torch_gc()

    shared.state.begin(job="extras")

    outputs = []

    if isinstance(image, dict):
        image = image["composite"]

    def get_images(extras_mode, image, image_folder, input_dir):
        if extras_mode == 1:
            for img in image_folder:
                if isinstance(img, Image.Image):
                    image = images.fix_image(img)
                    fn = ""
                else:
                    image = images.read(os.path.abspath(img.name))
                    fn = os.path.splitext(img.name)[0]
                yield image, fn
        elif extras_mode == 2:
            assert not shared.cmd_opts.hide_ui_dir_config, "--hide-ui-dir-config option must be disabled"
            assert input_dir, "input directory not selected"

            image_list = shared.listfiles(input_dir)
            for filename in image_list:
                yield filename, filename
        else:
            assert image, "image not selected"
            yield image, None

    if extras_mode == 2 and output_dir != "":
        outpath = output_dir
    else:
        outpath = opts.outdir_samples or opts.outdir_extras_samples

    infotext = ""
    comparison_html = ""

    data_to_process = list(get_images(extras_mode, image, image_folder, input_dir))
    shared.state.job_count = len(data_to_process)

    for image_placeholder, name in data_to_process:
        image_data: Image.Image

        shared.state.nextjob()
        shared.state.textinfo = name
        shared.state.skipped = False

        if shared.state.interrupted or shared.state.stopping_generation:
            break

        if isinstance(image_placeholder, str):
            try:
                image_data = images.read(image_placeholder)
            except Exception:
                continue
        else:
            image_data = image_placeholder

        image_data = image_data if image_data.mode in ("RGBA", "RGB") else image_data.convert("RGB")

        parameters, existing_pnginfo = images.read_info_from_image(image_data)
        if parameters:
            existing_pnginfo["parameters"] = parameters

        original_image = image_data.copy()
        initial_pp = scripts_postprocessing.PostprocessedImage(image_data)

        scripts.scripts_postproc.run(initial_pp, args)

        if shared.state.skipped:
            continue

        used_suffixes = {}
        for pp in [initial_pp, *initial_pp.extra_images]:
            suffix = pp.get_suffix(used_suffixes)

            if opts.use_original_name_batch and name is not None:
                basename = os.path.splitext(os.path.basename(name))[0]
                forced_filename = basename + suffix
            else:
                basename = ""
                forced_filename = None

            infotext = ", ".join([k if k == v else f"{k}: {infotext_utils.quote(v)}" for k, v in pp.info.items() if v is not None])

            if opts.enable_pnginfo:
                pp.image.info = existing_pnginfo

            shared.state.assign_current_image(pp.image)

            if save_output:
                fullfn, _ = images.save_image(pp.image, path=outpath, basename=basename, extension=opts.samples_format, info=infotext, short_filename=True, no_prompt=True, grid=False, pnginfo_section_name="postprocessing", existing_info=existing_pnginfo, forced_filename=forced_filename, suffix=suffix)

            if extras_mode != 2 or show_extras_results:
                outputs.append(pp.image)
                if not comparison_html:
                    comparison_html = _render_before_after_comparison(original_image, pp.image)

    devices.torch_gc()
    shared.state.end()
    return outputs, gr.update(value=None, visible=False), ui_common.plaintext_to_html(infotext), "", comparison_html


def run_postprocessing_video(_mode, _img, _folder, _in_dir, _out_dir, _show, video_input, *args, save_output: bool = True):
    devices.torch_gc()

    shared.state.begin(job="extras")

    outputs: list[np.ndarray] = []

    video_input = _normalize_video_input(video_input)
    assert video_input, "video not selected"

    container = av.open(video_input)

    video_stream = container.streams.best("video")
    frames = video_stream.frames

    def get_frames():
        for frame in tqdm(container.decode(video=0), desc="Processing Video", total=frames, unit="frame"):
            yield frame.to_image()

    infotext = None
    comparison_html = ""

    shared.state.job_count = frames

    for i, image_data in enumerate(get_frames()):

        shared.state.nextjob()
        shared.state.textinfo = str(i)
        shared.state.skipped = False

        if shared.state.interrupted or shared.state.stopping_generation:
            break

        original_image = image_data.copy()
        initial_pp = scripts_postprocessing.PostprocessedImage(image_data)

        scripts.scripts_postproc.run(initial_pp, args)

        if shared.state.skipped:
            continue

        if infotext is None:
            infotext = ", ".join([k if k == v else f"{k}: {infotext_utils.quote(v)}" for k, v in initial_pp.info.items() if v is not None])

        shared.state.assign_current_image(initial_pp.image)
        if not comparison_html:
            comparison_html = _render_before_after_comparison(original_image, initial_pp.image)

        outputs.append(np.array(initial_pp.image, dtype=np.uint8))

    if not (shared.state.interrupted or shared.state.stopping_generation):
        output_video = images.save_video(
            os.path.splitext(os.path.basename(video_input))[0],
            outputs,
            fps=round(float(container.streams.video[0].average_rate)),
            basename=None,
            info=infotext,
            audio_copy=video_input,
        )
    else:
        output_video = None

    container.close()
    devices.torch_gc()
    shared.state.end()
    video_comparison = _render_video_before_after_comparison(video_input, output_video) or comparison_html
    return outputs[-1:] if outputs else [], gr.update(value=output_video, visible=bool(output_video)), ui_common.plaintext_to_html(infotext), "", video_comparison


def run_postprocessing_webui(id_task, *args, **kwargs):
    if args[0] == 3:
        return run_postprocessing_video(*args, **kwargs)
    else:
        return run_postprocessing(*args, **kwargs)


def run_extras(extras_mode, resize_mode, image, image_folder, input_dir, output_dir, show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, upscale_first: bool, save_output: bool = True, max_side_length: int = 0):
    # Handler for API (does not support video)

    args = scripts.scripts_postproc.create_args_for_run(
        {
            "Upscale": {
                "upscale_enabled": True,
                "upscale_mode": resize_mode,
                "upscale_by": upscaling_resize,
                "max_side_length": max_side_length,
                "upscale_to_width": upscaling_resize_w,
                "upscale_to_height": upscaling_resize_h,
                "upscale_crop": upscaling_crop,
                "upscaler_1_name": extras_upscaler_1,
                "upscaler_2_name": extras_upscaler_2,
                "upscaler_2_visibility": extras_upscaler_2_visibility,
            },
            "GFPGAN": {
                "enable": True,
                "gfpgan_visibility": gfpgan_visibility,
            },
            "CodeFormer": {
                "enable": True,
                "codeformer_visibility": codeformer_visibility,
                "codeformer_weight": codeformer_weight,
            },
        }
    )

    result = run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, "", *args, save_output=save_output)
    return result[0], result[2], result[3]
