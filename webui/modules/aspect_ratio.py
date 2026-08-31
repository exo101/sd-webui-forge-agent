"""
Aspect ratio helper module (integrated from aspect-ratio-helper plugin).

Provides quick aspect ratio buttons for the txt2img / img2img dimension
controls. Buttons use Python callbacks that preserve total pixel count
and snap values to the slider step.
"""
import gradio as gr

from modules import shared
from modules.options import OptionInfo, options_section

# Aspect ratio presets: (display label, width ratio, height ratio)
ASPECT_RATIOS = [
    ("1:1", 1, 1),
    ("3:2", 3, 2),
    ("2:3", 2, 3),
    ("4:3", 4, 3),
    ("3:4", 3, 4),
    ("16:9", 16, 9),
    ("9:16", 9, 16),
    ("21:9", 21, 9),
    ("9:21", 9, 21),
]

# Slider bounds (match ui.py width/height slider min/max)
_MIN_DIM = 64
_MAX_DIM = 2048


def register_settings():
    """Register aspect ratio helper options into shared.opts."""
    opts = shared.opts
    # options_section assigns section + category_id so opts.reorder() can sort
    # these without crashing on None section. Placed under the "ui" category.
    items = options_section(
        ("aspect-ratio", "Aspect Ratio", "ui"),
        {
            "arh_show_aspect_buttons": OptionInfo(True, "Show aspect ratio quick buttons").needs_reload_ui(),
        },
    )
    for key, info in items.items():
        if key not in opts.data_labels:
            opts.add_option(key, info)


def _snap(value, step):
    """Snap *value* to the nearest multiple of *step*, clamped to slider bounds."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 1024.0
    v = round(v / step) * step
    return max(_MIN_DIM, min(_MAX_DIM, int(v)))


def _apply_ratio(width, height, w_ratio, h_ratio, step=8):
    """Return (new_width, new_height) for the given aspect ratio.

    Total pixel count is preserved (approx) so the change feels natural
    instead of jumping to a fixed resolution.
    """
    if not width or not height:
        width = height = 1024
    pixels = float(width) * float(height)
    ratio = w_ratio / h_ratio
    new_w = (pixels * ratio) ** 0.5
    new_h = new_w / ratio
    return _snap(new_w, step), _snap(new_h, step)


def create_aspect_ratio_buttons(tabname, width, height):
    """Render a row of aspect ratio quick buttons bound to *width* / *height* sliders."""
    with gr.Row(
        elem_id=f"{tabname}_ar_buttons",
        elem_classes=["aspect-ratio-buttons"],
        equal_height=True,
    ):
        for label, wr, hr in ASPECT_RATIOS:
            elem_id = f"{tabname}_ar_{label.replace(':', '')}"
            btn = gr.Button(value=label, elem_id=elem_id, size="sm")
            btn.click(
                fn=lambda w, h, wr=wr, hr=hr: _apply_ratio(w, h, wr, hr),
                inputs=[width, height],
                outputs=[width, height],
                show_progress=False,
                queue=False,
            )
