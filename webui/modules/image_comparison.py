"""
Image comparison helper module.

Powers the image comparison panel in the Extras tab:
- ``img2info`` extracts generation parameters from two PIL images and
  renders them as two HTML columns (A / B) for side-by-side comparison.
- ``_gjs`` returns the kwargs needed to attach a pure-JS click handler
  (no Python round-trip) to a Gradio component.
- ``register_settings`` registers any related options (currently none).
"""
import html as _html

from modules import images
from modules.ui_common import plaintext_to_html


def _extract_geninfo(image):
    """Return the generation parameters string embedded in *image*, or ''."""
    if image is None:
        return ""
    try:
        geninfo, _items = images.read_info_from_image(image)
    except Exception:
        return ""
    return geninfo or ""


def _render_info(label, geninfo):
    """Render an HTML block showing the generation parameters for one image."""
    safe_label = _html.escape(label)
    if not geninfo:
        return (
            f'<div class="img-comp-info-block">'
            f'<div class="img-comp-info-title">{safe_label}</div>'
            f'<div class="img-comp-info-empty">No generation info</div>'
            f"</div>"
        )
    # Each line of generation info is shown as a row; escape to avoid XSS.
    lines = [plaintext_to_html(line) for line in str(geninfo).splitlines() if line.strip()]
    body = "<br>".join(lines)
    return (
        f'<div class="img-comp-info-block">'
        f'<div class="img-comp-info-title">{safe_label}</div>'
        f'<div class="img-comp-info-body">{body}</div>'
        f"</div>"
    )


def img2info(image_a, image_b):
    """Extract generation info from both images and return two HTML columns."""
    info_a = _render_info("Image A", _extract_geninfo(image_a))
    info_b = _render_info("Image B", _extract_geninfo(image_b))
    return info_a, info_b


def _gjs(js_code):
    """Return kwargs for a Gradio click that runs *js_code* client-side only."""
    # NOTE: call sites already pass fn=None explicitly, so we must NOT include
    # "fn" here — otherwise Gradio raises "got multiple values for keyword
    # argument 'fn'".
    return {"_js": js_code, "show_progress": False, "queue": False}


def register_settings():
    """Register image-comparison-related options (reserved for future use)."""
    # No options required currently; kept for API parity with ui.py call site.
    pass
