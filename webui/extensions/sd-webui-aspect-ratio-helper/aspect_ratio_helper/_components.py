from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from functools import partial
from typing import Callable

import gradio as gr

import aspect_ratio_helper._constants as _constants
import aspect_ratio_helper._settings as _settings
import aspect_ratio_helper._util as _util


class ArhUIComponent(ABC):

    def __init__(self, script):
        self.script = script

    @abstractmethod
    def render(self): ...

    @staticmethod
    @abstractmethod
    def should_show() -> bool: ...

    @staticmethod
    @abstractmethod
    def add_options(): ...


class MaxDimensionScaler(ArhUIComponent):

    def render(self):
        max_dim_default = _settings.safe_opt(
            _constants.ARH_MAX_WIDTH_OR_HEIGHT_KEY,
        )
        self.script.max_dimension = float(max_dim_default)

        inputs = outputs = [self.script.wc, self.script.hc]

        with gr.Row(
                visible=self.should_show(),
        ):
            max_dim_default = _settings.safe_opt(
                _constants.ARH_MAX_WIDTH_OR_HEIGHT_KEY,
            )
            max_dimension_slider = gr.Slider(
                minimum=_constants.MIN_DIMENSION,
                maximum=_constants.MAX_DIMENSION,
                step=8,
                value=max_dim_default,
                label='Maximum dimension',
            )

            def _update_max_dimension(_max_dimension):
                self.script.max_dimension = _max_dimension

            max_dimension_slider.change(
                _update_max_dimension,
                inputs=[max_dimension_slider],
                show_progress=False,
            )

            gr.Button(
                value='Scale to maximum dimension',
                visible=self.should_show(),
            ).click(
                fn=_util.scale_dimensions_to_max_dim,
                inputs=[*inputs, max_dimension_slider],
                outputs=outputs,
            )

    @staticmethod
    def should_show() -> bool:
        return _settings.safe_opt(_constants.ARH_SHOW_MAX_WIDTH_OR_HEIGHT_KEY)

    @staticmethod
    def add_options(shared):
        shared.opts.add_option(
            key=_constants.ARH_SHOW_MAX_WIDTH_OR_HEIGHT_KEY,
            info=shared.OptionInfo(
                default=_settings.OPT_KEY_TO_DEFAULT_MAP.get(
                    _constants.ARH_SHOW_MAX_WIDTH_OR_HEIGHT_KEY,
                ),
                label='Show maximum dimension button',
                section=_constants.SECTION,
            ),
        )
        shared.opts.add_option(
            key=_constants.ARH_MAX_WIDTH_OR_HEIGHT_KEY,
            info=shared.OptionInfo(
                default=_settings.OPT_KEY_TO_DEFAULT_MAP.get(
                    _constants.ARH_MAX_WIDTH_OR_HEIGHT_KEY,
                ),
                label='Maximum dimension default',
                component=gr.Slider,
                component_args={
                    'minimum': _constants.MIN_DIMENSION,
                    'maximum': _constants.MAX_DIMENSION,
                    'step': 8,
                },
                section=_constants.SECTION,
            ),
        )


class MinDimensionScaler(ArhUIComponent):

    def render(self):
        min_dim_default = _settings.safe_opt(
            _constants.ARH_MIN_WIDTH_OR_HEIGHT_KEY,
        )
        self.script.min_dimension = float(min_dim_default)

        inputs = outputs = [self.script.wc, self.script.hc]

        with gr.Row(
                visible=self.should_show(),
        ):
            min_dim_default = _settings.safe_opt(
                _constants.ARH_MIN_WIDTH_OR_HEIGHT_KEY,
            )
            min_dimension_slider = gr.Slider(
                minimum=_constants.MIN_DIMENSION,
                maximum=_constants.MAX_DIMENSION,
                step=8,
                value=min_dim_default,
                label='Minimum dimension',
            )

            def _update_min_dimension(_min_dimension):
                self.script.min_dimension = _min_dimension

            min_dimension_slider.change(
                _update_min_dimension,
                inputs=[min_dimension_slider],
                show_progress=False,
            )

            gr.Button(
                value='Scale to minimum dimension',
                visible=self.should_show(),
            ).click(
                fn=_util.scale_dimensions_to_min_dim,
                inputs=[*inputs, min_dimension_slider],
                outputs=outputs,
            )

    @staticmethod
    def should_show() -> bool:
        return _settings.safe_opt(_constants.ARH_SHOW_MIN_WIDTH_OR_HEIGHT_KEY)

    @staticmethod
    def add_options(shared):
        shared.opts.add_option(
            key=_constants.ARH_SHOW_MIN_WIDTH_OR_HEIGHT_KEY,
            info=shared.OptionInfo(
                default=_settings.OPT_KEY_TO_DEFAULT_MAP.get(
                    _constants.ARH_SHOW_MIN_WIDTH_OR_HEIGHT_KEY,
                ),
                label='Show minimum dimension button',
                section=_constants.SECTION,
            ),
        )
        shared.opts.add_option(
            key=_constants.ARH_MIN_WIDTH_OR_HEIGHT_KEY,
            info=shared.OptionInfo(
                default=_settings.OPT_KEY_TO_DEFAULT_MAP.get(
                    _constants.ARH_MIN_WIDTH_OR_HEIGHT_KEY,
                ),
                label='Minimum dimension default',
                component=gr.Slider,
                component_args={
                    'minimum': _constants.MIN_DIMENSION,
                    'maximum': _constants.MAX_DIMENSION,
                    'step': 8,
                },
                section=_constants.SECTION,
            ),
        )


class PredefinedAspectRatioButtons(ArhUIComponent):

    def render(self):
        use_max_dim_op = _settings.safe_opt(
            _constants.ARH_PREDEFINED_ASPECT_RATIO_USE_MAX_DIM_KEY,
        )
        aspect_ratios = _settings.safe_opt(
            _constants.ARH_PREDEFINED_ASPECT_RATIOS_KEY,
        ).split(',')

        with gr.Column(
            variant='panel',
            visible=self.should_show(),
        ):
            # Build HTML for graphical aspect ratio shape buttons
            html = '<div class="arh-shape-buttons">'
            for ar_str in aspect_ratios:
                ar_str = ar_str.strip()
                w, h, *_ = (abs(float(d)) for d in ar_str.split(':'))
                # Calculate relative dimensions for the shape
                max_dim = max(w, h)
                shape_w = int(36 * w / max_dim)
                shape_h = int(36 * h / max_dim)
                shape_w = max(shape_w, 16)
                shape_h = max(shape_h, 16)
                html += f'''
                <div class="arh-shape-btn" data-ar="{ar_str}">
                    <div class="arh-shape-rect" style="width:{shape_w}px;height:{shape_h}px;"></div>
                    <span class="arh-shape-label">{ar_str}</span>
                </div>
                '''
            html += '</div>'
            gr.HTML(html)

        # Hidden buttons to trigger the actual Python functions
        with gr.Row(visible=False):
            for ar_str in aspect_ratios:
                ar_str = ar_str.strip()
                w, h, *_ = (abs(float(d)) for d in ar_str.split(':'))

                inputs = []
                if use_max_dim_op:
                    ar_func = partial(
                        _util.scale_dimensions_to_max_dim_func,
                        width=w, height=h,
                        max_dim=lambda: self.script.max_dimension,
                    )
                else:
                    inputs.extend([self.script.wc, self.script.hc])
                    ar_func = partial(
                        _util.scale_dimensions_to_ui_width_or_height,
                        arw=w, arh=h,
                    )

                hidden_btn = gr.Button(
                    value=f'ar_{ar_str}',
                    elem_id=f'arh_hidden_btn_{ar_str.replace(":", "_")}',
                    visible=False,
                )
                hidden_btn.click(
                    fn=ar_func,
                    inputs=inputs,
                    outputs=[self.script.wc, self.script.hc],
                )

    @staticmethod
    def should_show() -> bool:
        return _settings.safe_opt(
            _constants.ARH_SHOW_PREDEFINED_ASPECT_RATIOS_KEY,
        )

    @staticmethod
    def add_options(shared):
        shared.opts.add_option(
            key=_constants.ARH_SHOW_PREDEFINED_ASPECT_RATIOS_KEY,
            info=shared.OptionInfo(
                default=_settings.OPT_KEY_TO_DEFAULT_MAP.get(
                    _constants.ARH_SHOW_PREDEFINED_ASPECT_RATIOS_KEY,
                ),
                label='Show pre-defined aspect ratio buttons',
                section=_constants.SECTION,
            ),
        )
        shared.opts.add_option(
            key=_constants.ARH_PREDEFINED_ASPECT_RATIO_USE_MAX_DIM_KEY,
            info=shared.OptionInfo(
                default=_settings.OPT_KEY_TO_DEFAULT_MAP.get(
                    _constants.ARH_PREDEFINED_ASPECT_RATIO_USE_MAX_DIM_KEY,
                ),
                label='Use "Maximum dimension" for aspect ratio buttons (by '
                      'default we use the max width or height)',
                section=_constants.SECTION,
            ),
        )
        shared.opts.add_option(
            key=_constants.ARH_PREDEFINED_ASPECT_RATIOS_KEY,
            info=shared.OptionInfo(
                default=_settings.OPT_KEY_TO_DEFAULT_MAP.get(
                    _constants.ARH_PREDEFINED_ASPECT_RATIOS_KEY,
                ),
                label='Pre-defined aspect ratio buttons '
                      '(1:1, 4:3, 16:9, 9:16, 21:9)',
                section=_constants.SECTION,
            ),
        )

    @property
    def display_func(self) -> Callable[[str], str]:
        return lambda _: None  # todo: different displays for aspect ratios.


class PredefinedPercentageButtons(ArhUIComponent):

    def render(self):
        inputs = outputs = [self.script.wc, self.script.hc]
        with gr.Column(
                variant='panel',
                visible=self.should_show(),
        ):
            with gr.Row(
                variant='compact',
                visible=self.should_show(),
            ):
                pct_slider = gr.Slider(
                    minimum=10,
                    maximum=200,
                    step=5,
                    value=100,
                    label='缩放比例 (%)',
                    visible=self.should_show(),
                )
                apply_btn = gr.Button(
                    value='应用缩放',
                    scale=0,
                    min_width=80,
                    visible=self.should_show(),
                )

                def _apply_scale(width, height, pct_val):
                    return _util.scale_by_percentage(width, height, pct_val / 100)

                apply_btn.click(
                    fn=_apply_scale,
                    inputs=[self.script.wc, self.script.hc, pct_slider],
                    outputs=outputs,
                )

    @staticmethod
    def should_show() -> bool:
        return _settings.safe_opt(
            _constants.ARH_SHOW_PREDEFINED_PERCENTAGES_KEY,
        )

    @staticmethod
    def add_options(shared):
        shared.opts.add_option(
            key=_constants.ARH_SHOW_PREDEFINED_PERCENTAGES_KEY,
            info=shared.OptionInfo(
                default=_settings.OPT_KEY_TO_DEFAULT_MAP.get(
                    _constants.ARH_SHOW_PREDEFINED_PERCENTAGES_KEY,
                ),
                label='Show pre-defined percentage buttons',
                section=_constants.SECTION,
            ),
        )
        shared.opts.add_option(
            key=_constants.ARH_PREDEFINED_PERCENTAGES_KEY,
            info=shared.OptionInfo(
                default=_settings.OPT_KEY_TO_DEFAULT_MAP.get(
                    _constants.ARH_PREDEFINED_PERCENTAGES_KEY,
                ),
                label='Pre-defined percentage buttons (75, 125, 150)',
                section=_constants.SECTION,
            ),
        )
        shared.opts.add_option(
            key=_constants.ARH_PREDEFINED_PERCENTAGES_DISPLAY_KEY,
            info=shared.OptionInfo(
                default=_settings.OPT_KEY_TO_DEFAULT_MAP.get(
                    _constants.ARH_PREDEFINED_PERCENTAGES_DISPLAY_KEY,
                ),
                label='Pre-defined percentage display format',
                component=gr.Dropdown,
                component_args=lambda: {
                    'choices': tuple(
                        _settings.PREDEFINED_PERCENTAGES_DISPLAY_MAP.keys(),
                    ),
                },
                section=_constants.SECTION,
            ),
        )

    @property
    def display_func(self) -> Callable[[str], str]:
        return _settings.PREDEFINED_PERCENTAGES_DISPLAY_MAP.get(
            _settings.safe_opt(
                _constants.ARH_PREDEFINED_PERCENTAGES_DISPLAY_KEY,
            ),
        )
