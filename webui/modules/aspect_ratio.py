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
    ("16:9", 16, 9),
    ("9:16", 9, 16),
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


def _apply_scale(scale, base, width, height, step=8):
    """Return (new_width, new_height, new_base) for a percentage scale.

    Uses *base* (a (w, h) list stored in gr.State) so that moving the
    slider does not compound on previous scaled values. When scale is 100%
    it restores FROM the stored base; the base itself is only updated via
    the width/height change callbacks when the user manually edits at 100%.
    """
    if scale is None:
        scale = 100
    base_w = base[0] if base else None
    base_h = base[1] if base else None
    # First run: capture current dimensions as the base
    if base_w is None or base_h is None:
        base_w = _snap(width, step) if width else 1024
        base_h = _snap(height, step) if height else 1024
        return base_w, base_h, [base_w, base_h]
    # At 100%: restore original base dimensions
    if scale == 100:
        return base_w, base_h, [base_w, base_h]
    # Otherwise: scale from the base
    new_w = _snap(base_w * scale / 100.0, step)
    new_h = _snap(base_h * scale / 100.0, step)
    return new_w, new_h, [base_w, base_h]


def _update_base_on_manual_edit(scale, base, width, height, step=8):
    """Update the stored base when the user manually edits width/height
    while the scale slider is at 100% (i.e. not actively scaling).

    When scale != 100, returns the existing *base* unchanged so that
    programmatic width changes from scale adjustments don't corrupt it.
    """
    if scale is None or scale == 100:
        w = _snap(width, step) if width else 1024
        h = _snap(height, step) if height else 1024
        return [w, h]
    return base  # keep existing base unchanged


def create_size_scale_slider(tabname, width, height):
    """Render a percentage slider (10-200%) to scale width/height proportionally.

    Uses gr.State to store the base dimensions so scaling is non-destructive
    and does not compound.
    """
    base_state = gr.State(value=None)
    with gr.Row(elem_id=f"{tabname}_scale_row", elem_classes=["size-scale-row"]):
        scale = gr.Slider(
            minimum=10,
            maximum=200,
            step=5,
            value=100,
            label="尺寸 %",
            elem_id=f"{tabname}_size_scale",
        )
        scale.change(
            fn=_apply_scale,
            inputs=[scale, base_state, width, height],
            outputs=[width, height, base_state],
            show_progress=False,
            queue=False,
        )
    # When user manually edits width/height at 100%, refresh the base.
    # base_state is passed as input so we can return it unchanged when
    # scale != 100 (avoiding corruption from programmatic width changes).
    width.change(
        fn=_update_base_on_manual_edit,
        inputs=[scale, base_state, width, height],
        outputs=[base_state],
        show_progress=False,
        queue=False,
    )
    height.change(
        fn=_update_base_on_manual_edit,
        inputs=[scale, base_state, width, height],
        outputs=[base_state],
        show_progress=False,
        queue=False,
    )
