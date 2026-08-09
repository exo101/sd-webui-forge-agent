import gradio as gr

def create_quick_description(chat_message):
    """创建快捷描述功能组件"""
    
    def get_caption_prompt(caption_type):
        """获取不同类型的图像描述提示词模板"""
        prompts = {
            "descriptive_formal": "Write a detailed description for this image in a formal tone.",
            "mj_prompt": "Write a MidJourney prompt for this image.",
            "storyboard_description": "Analyze this image as a storyboard panel. Describe the composition, framing, camera angle, character positioning, and visual storytelling elements. Include details about panel transitions, gutters, and how this panel fits into a larger narrative sequence.",
            "video_description": "Based on this image, create a detailed description for a video that starts with this scene. Describe the initial setting, characters, mood, and potential movements or actions that could happen in the following scenes.",
            "text_video_description": "Create a detailed text-to-video generation prompt based on this image. Describe the complete scene including environment, characters, actions, camera movements, transitions, and visual effects that would make a compelling video.",
            "art_critic": "简单描述图像."
        }
        return prompts.get(caption_type, "")

    # 定义所有按钮配置 - 按功能分组
    all_button_rows = [
        [
            ("自然语言描述文本", "descriptive_formal"),
            ("MidJourney提示词", "mj_prompt"),
            ("分镜构图描述", "storyboard_description")
        ],
        [
            ("图生视频描述", "video_description"),
            ("文生视频描述文本", "text_video_description"),
            ("简单描述", "art_critic")
        ]
    ]
    
    # 创建按钮网格布局
    buttons = []
    with gr.Column():
        for button_row in all_button_rows:
            with gr.Row():
                for label, caption_type in button_row:
                    if label and caption_type:  # 只创建非空按钮
                        btn = gr.Button(label)
                        btn.click(
                            fn=lambda ct=caption_type: get_caption_prompt(ct),
                            inputs=None,
                            outputs=chat_message
                        )
                        buttons.append(btn)
                    else:
                        # 添加空占位符以保持布局
                        with gr.Column(scale=1):
                            gr.Markdown("")  # 空白占位
    
    return buttons


def optimize_sd_prompt_tags(tags, max_tags=255):
    """
    简化版的Stable Diffusion提示词标签优化函数
    只进行去重处理，限制标签数量不超过255个
    
    Args:
        tags: 图像识别生成的标签列表或字符串
        max_tags: 最大标签数量，默认255个
        
    Returns:
        优化后的标签字符串
    """
    # 直接处理输入，只去重并限制数量
    if isinstance(tags, str):
        # 如果是字符串，按逗号分割并去除空白
        tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
    else:
        # 如果是列表，直接转换为列表
        tag_list = list(tags)
    
    # 1. 去重
    tag_list = list(set(tag_list))
    
    # 2. 限制标签数量
    if len(tag_list) > max_tags:
        tag_list = tag_list[:max_tags]
    
    # 3. 重新组合成逗号分隔的字符串
    return ', '.join(tag_list)