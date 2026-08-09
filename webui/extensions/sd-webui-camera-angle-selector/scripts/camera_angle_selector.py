import gradio as gr
import os
import urllib.parse
from modules import script_callbacks
from pathlib import Path


def get_html_path():
    return str(Path(__file__).parent / "camera_3d_view.html")


def create_angle_selector_ui():
    html_path = get_html_path()
    
    with gr.Blocks() as ui:
        gr.Markdown("## 多角度提示词可视化选择器")
        
        with gr.Row():
            gr.HTML(f'''
            <div style="position:relative; width:100%; height:500px;">
                <iframe id="camera-iframe" name="camera-iframe" 
                        src="/file={urllib.parse.quote(html_path)}" 
                        width="100%" height="100%" 
                        style="border: 1px solid #444; border-radius: 8px;"></iframe>
            </div>
            ''')
        
        with gr.Row():
            apply_to_txt2img_btn = gr.Button("应用到文生图", variant="primary")
            apply_to_img2img_btn = gr.Button("应用到图生图", variant="primary")
        
        gr.Markdown("""
        ### 使用说明：
        1. 拖拽红色手柄调整方位角（正面/侧面/背面等）
        2. 拖拽绿色手柄调整高程角（仰视/平视/俯视）
        3. 拖拽黄色手柄调整距离（特写/中景/远景）
        4. 点击"应用到文生图"或"应用到图生图"按钮将角度提示词添加到对应的输入框
        """)
        
        apply_js_txt2img = """
        async () => {
            const iframe = document.getElementById('camera-iframe');
            
            if (!iframe || !iframe.contentWindow) {
                alert('错误：无法访问3D视角界面，请刷新页面后重试！');
                return null;
            }
            
            await new Promise(resolve => {
                if (iframe.contentDocument && iframe.contentDocument.readyState === 'complete') {
                    resolve();
                } else {
                    iframe.onload = resolve;
                    setTimeout(resolve, 1000);
                }
            });
            
            iframe.contentWindow.postMessage({ type: 'GET_CURRENT_ANGLE' }, '*');
            
            return new Promise((resolve) => {
                const startTime = Date.now();
                const timeoutId = setTimeout(() => {
                    window.removeEventListener('message', handleMessage);
                    alert('错误：等待3D视角界面响应超时！');
                    resolve(null);
                }, 3000);
                
                const handleMessage = (event) => {
                    if (event.data.type === 'ANGLE_SELECTED') {
                        clearTimeout(timeoutId);
                        window.removeEventListener('message', handleMessage);
                        
                        const parts = [];
                        if (event.data.azimuth && event.data.azimuth !== "") parts.push(event.data.azimuth);
                        if (event.data.elevation && event.data.elevation !== "") parts.push(event.data.elevation);
                        if (event.data.distance && event.data.distance !== "") parts.push(event.data.distance);
                        
                        if (parts.length === 0) {
                            alert('错误：未能获取到有效的角度数据！');
                            resolve(null);
                            return;
                        }
                        
                        const anglePrompt = parts.join(' ');
                        
                        const element = document.getElementById('txt2img_prompt');
                        if (element) {
                            let textarea = element;
                            if (element.tagName !== 'TEXTAREA' && element.tagName !== 'INPUT') {
                                textarea = element.querySelector('textarea') || element.querySelector('input');
                            }
                            if (textarea) {
                                if (textarea.value) {
                                    textarea.value = textarea.value + ' ' + anglePrompt;
                                } else {
                                    textarea.value = anglePrompt;
                                }
                                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                                textarea.dispatchEvent(new Event('change', { bubbles: true }));
                                alert('已应用到文生图：' + anglePrompt);
                            }
                        } else {
                            alert('未找到文生图提示词输入框，请确保已切换到文生图标签页！');
                        }
                        
                        resolve(anglePrompt);
                    }
                };
                
                window.addEventListener('message', handleMessage);
            });
        }
        """
        
        apply_js_img2img = """
        async () => {
            const iframe = document.getElementById('camera-iframe');
            
            if (!iframe || !iframe.contentWindow) {
                alert('错误：无法访问3D视角界面，请刷新页面后重试！');
                return null;
            }
            
            await new Promise(resolve => {
                if (iframe.contentDocument && iframe.contentDocument.readyState === 'complete') {
                    resolve();
                } else {
                    iframe.onload = resolve;
                    setTimeout(resolve, 1000);
                }
            });
            
            iframe.contentWindow.postMessage({ type: 'GET_CURRENT_ANGLE' }, '*');
            
            return new Promise((resolve) => {
                const startTime = Date.now();
                const timeoutId = setTimeout(() => {
                    window.removeEventListener('message', handleMessage);
                    alert('错误：等待3D视角界面响应超时！');
                    resolve(null);
                }, 3000);
                
                const handleMessage = (event) => {
                    if (event.data.type === 'ANGLE_SELECTED') {
                        clearTimeout(timeoutId);
                        window.removeEventListener('message', handleMessage);
                        
                        const parts = [];
                        if (event.data.azimuth && event.data.azimuth !== "") parts.push(event.data.azimuth);
                        if (event.data.elevation && event.data.elevation !== "") parts.push(event.data.elevation);
                        if (event.data.distance && event.data.distance !== "") parts.push(event.data.distance);
                        
                        if (parts.length === 0) {
                            alert('错误：未能获取到有效的角度数据！');
                            resolve(null);
                            return;
                        }
                        
                        const anglePrompt = parts.join(' ');
                        
                        const element = document.getElementById('img2img_prompt');
                        if (element) {
                            let textarea = element;
                            if (element.tagName !== 'TEXTAREA' && element.tagName !== 'INPUT') {
                                textarea = element.querySelector('textarea') || element.querySelector('input');
                            }
                            if (textarea) {
                                if (textarea.value) {
                                    textarea.value = textarea.value + ' ' + anglePrompt;
                                } else {
                                    textarea.value = anglePrompt;
                                }
                                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                                textarea.dispatchEvent(new Event('change', { bubbles: true }));
                                alert('已应用到图生图：' + anglePrompt);
                            }
                        } else {
                            alert('未找到图生图提示词输入框，请确保已切换到图生图标签页！');
                        }
                        
                        resolve(anglePrompt);
                    }
                };
                
                window.addEventListener('message', handleMessage);
            });
        }
        """
        
        dummy_output = gr.Textbox(visible=False)
        
        apply_to_txt2img_btn.click(
            fn=None,
            inputs=[],
            outputs=[dummy_output],
            _js=apply_js_txt2img
        )
        
        apply_to_img2img_btn.click(
            fn=None,
            inputs=[],
            outputs=[dummy_output],
            _js=apply_js_img2img
        )
    
    return [(ui, "相机角度选择器", "camera_angle_selector")]


script_callbacks.on_ui_tabs(create_angle_selector_ui)
