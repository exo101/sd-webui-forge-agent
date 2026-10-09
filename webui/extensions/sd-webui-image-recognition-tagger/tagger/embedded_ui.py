"""WD14 UI embedded inside the image recognition tab."""
from .ui import *

def create_embedded_ui():
        with gr.Row():
            with gr.Column(variant='panel'):
    
                # input components
                with gr.Tabs():
                    with gr.TabItem(label='Single process'):
                        image = gr.Image(
                            label='Source',
                            source='upload',
                            interactive=True,
                            type="pil"
                        )
    
                    with gr.TabItem(label='Batch from directory'):
                        batch_input_glob = utils.preset.component(
                            gr.Textbox,
                            label='Input directory',
                            placeholder='/path/to/images or /path/to/images/**/*'
                        )
                        batch_input_recursive = utils.preset.component(
                            gr.Checkbox,
                            label='Use recursive with glob pattern'
                        )
    
                        batch_output_dir = utils.preset.component(
                            gr.Textbox,
                            label='Output directory',
                            placeholder='Leave blank to save images to the same path.'
                        )
    
                        batch_output_filename_format = utils.preset.component(
                            gr.Textbox,
                            label='Output filename format',
                            placeholder='Leave blank to use same filename as original.',
                            value='[name].[output_extension]'
                        )
    
                        import hashlib
                        with gr.Accordion(
                            label='Output filename formats',
                            open=False
                        ):
                            gr.Markdown(
                                value=f'''
                                ### Related to original file
                                - `[name]`: Original filename without extension
                                - `[extension]`: Original extension
                                - `[hash:<algorithms>]`: Original extension
                                    Available algorithms: `{', '.join(hashlib.algorithms_available)}`
    
                                ### Related to output file
                                - `[output_extension]`: Output extension (has no dot)
    
                                ## Examples
                                ### Original filename without extension
                                `[name].[output_extension]`
    
                                ### Original file's hash (good for deleting duplication)
                                `[hash:sha1].[output_extension]`
                                '''
                            )
    
                        batch_output_action_on_conflict = utils.preset.component(
                            gr.Dropdown,
                            label='Action on existing caption',
                            value='ignore',
                            choices=[
                                'ignore',
                                'copy',
                                'append',
                                'prepend'
                            ]
                        )
    
                        batch_remove_duplicated_tag = utils.preset.component(
                            gr.Checkbox,
                            label='Remove duplicated tag'
                        )
    
                        batch_output_save_json = utils.preset.component(
                            gr.Checkbox,
                            label='Save with JSON'
                        )
    
                submit = gr.Button(
                    value='Interrogate',
                    variant='primary'
                )
    
                info = gr.HTML()
    
                # preset selector
                with gr.Row(variant='compact'):
                    available_presets = utils.preset.list()
                    selected_preset = gr.Dropdown(
                        label='Preset',
                        choices=available_presets,
                        value=available_presets[0]
                    )
    
                    save_preset_button = gr.Button(
                        value=ui.save_style_symbol
                    )
    
                    ui.create_refresh_button(
                        selected_preset,
                        lambda: None,
                        lambda: {'choices': utils.preset.list()},
                        'refresh_preset'
                    )
    
                # option components
    
                # interrogator selector
                with gr.Column():
                    with gr.Row(variant='compact'):
                        interrogator_names = utils.refresh_interrogators()
                        interrogator = utils.preset.component(
                            gr.Dropdown,
                            label='Interrogator',
                            choices=interrogator_names,
                            value=(
                                None
                                if len(interrogator_names) < 1 else
                                interrogator_names[-1]
                            )
                        )
    
                        ui.create_refresh_button(
                            interrogator,
                            lambda: None,
                            lambda: {'choices': utils.refresh_interrogators()},
                            'refresh_interrogator'
                        )
    
                    unload_all_models = gr.Button(
                        value='Unload all interrogate models'
                    )
    
                threshold = utils.preset.component(
                    gr.Slider,
                    label='Threshold',
                    minimum=0,
                    maximum=1,
                    value=0.35
                )
    
                additional_tags = utils.preset.component(
                    gr.Textbox,
                    label='Additional tags (split by comma)',
                    elem_id='additioanl-tags'
                )
    
                exclude_tags = utils.preset.component(
                    gr.Textbox,
                    label='Exclude tags (split by comma)',
                    elem_id='exclude-tags'
                )
    
                sort_by_alphabetical_order = utils.preset.component(
                    gr.Checkbox,
                    label='Sort by alphabetical order',
                )
                add_confident_as_weight = utils.preset.component(
                    gr.Checkbox,
                    label='Include confident of tags matches in results'
                )
                replace_underscore = utils.preset.component(
                    gr.Checkbox,
                    label='Use spaces instead of underscore',
                    value=True
                )
                replace_underscore_excludes = utils.preset.component(
                    gr.Textbox,
                    label='Excudes (split by comma)',
                    # kaomoji from WD 1.4 tagger csv. thanks, Meow-San#5400!
                    value='0_0, (o)_(o), +_+, +_-, ._., <o>_<o>, <|>_<|>, =_=, >_<, 3_3, 6_9, >_o, @_@, ^_^, o_o, u_u, x_x, |_|, ||_||'
                )
                escape_tag = utils.preset.component(
                    gr.Checkbox,
                    label='Escape brackets',
                )
    
                unload_model_after_running = utils.preset.component(
                    gr.Checkbox,
                    label='Unload model after running',
                )
    
            # output components
            with gr.Column(variant='panel'):
                tags = gr.Textbox(
                    label='Tags',
                    placeholder='Found tags',
                    interactive=False
                )
    
                with gr.Row():
                    parameters_copypaste.bind_buttons(
                        parameters_copypaste.create_buttons(
                            ["txt2img", "img2img"],
                        ),
                        None,
                        tags
                    )
    
                rating_confidents = gr.Label(
                    label='Rating confidents',
                    elem_id='rating-confidents'
                )
                tag_confidents = gr.Label(
                    label='Tag confidents',
                    elem_id='tag-confidents'
                )
    
        # register events
        selected_preset.change(
            fn=utils.preset.apply,
            inputs=[selected_preset],
            outputs=[*utils.preset.components, info]
        )
    
        save_preset_button.click(
            fn=utils.preset.save,
            inputs=[selected_preset, *utils.preset.components],  # values only
            outputs=[info]
        )
    
        unload_all_models.click(
            fn=unload_interrogators,
            outputs=[info]
        )
    
        for func in [image.change, submit.click]:
            func(
                # WD1.4 is an independent ONNX task. Do not put it behind
                # Forge's global GPU generation queue: doing so freezes the
                # generation controls while tagging is running.
                fn=on_interrogate,
                inputs=[
                    # single process
                    image,
    
                    # batch process
                    batch_input_glob,
                    batch_input_recursive,
                    batch_output_dir,
                    batch_output_filename_format,
                    batch_output_action_on_conflict,
                    batch_remove_duplicated_tag,
                    batch_output_save_json,
    
                    # options
                    interrogator,
                    threshold,
                    additional_tags,
                    exclude_tags,
                    sort_by_alphabetical_order,
                    add_confident_as_weight,
                    replace_underscore,
                    replace_underscore_excludes,
                    escape_tag,
    
                    unload_model_after_running
                ],
                outputs=[
                    tags,
                    rating_confidents,
                    tag_confidents,
    
                    # contains execution time, memory usage and other stuffs...
                    # generated from modules.ui.wrap_gradio_call
                    info
                ]
            )
    
