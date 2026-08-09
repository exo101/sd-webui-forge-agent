import gradio as gr
import os

def create_tag_management_module():
    """创建标签管理模块并返回组件结构"""
    result = {}

    # 添加Accordion包装，与其他模块保持一致
    with gr.Accordion("标签管理", open=False):
        with gr.Group():
            folder_path = gr.Textbox(label="文件夹路径", placeholder="输入包含txt文件的文件夹路径")
            
            # 定义文件下拉选择框
            selected_file = gr.Dropdown(label="选择文件", choices=[], interactive=True)
            # 使用已定义的selected_file作为文件列表
            file_list = selected_file
            
            # 添加文件路径文本框作为内部变量使用
            file_path = gr.Textbox(label="文件路径", visible=False)
            
            keyword_input = gr.Textbox(label="关键词", placeholder="输入要添加/删除的关键词")
            # 修改为支持取消选择的位置选择框
            position = gr.Radio(["开头", "结尾", "随机位置"], label="添加位置", value=None, interactive=True)
            
            with gr.Row():
                add_btn = gr.Button("添加关键词")
                remove_btn = gr.Button("删除关键词")
                
            # 恢复批量操作按钮
            with gr.Row():
                add_to_all_btn = gr.Button("对全部添加关键词")
                remove_from_all_btn = gr.Button("对全部删除关键词")
            
            result_output = gr.Textbox(label="操作结果", lines=5)
            
            def update_folder_path(path):
                if path and os.path.isdir(path):
                    # 查找文件夹中的txt文件
                    txt_files = [f for f in os.listdir(path) if f.endswith('.txt')]
                    # 返回更新后的路径和下拉选项
                    return path, gr.update(choices=txt_files), gr.update(choices=txt_files), path
                return folder_path, gr.update(choices=[]), gr.update(choices=[]), ""

            def update_selected_file(file_name, folder_path):
                if file_name and folder_path:
                    full_path = os.path.join(folder_path, file_name)
                    return full_path
                return ""

            def add_keyword(file_path, keyword, position):
                if not file_path or not keyword:
                    # 添加详细错误信息，区分路径和关键词
                    if not file_path and not keyword:
                        return "请提供文件路径和关键词"
                    elif not file_path:
                        return "请提供有效的文件路径"
                    else:
                        return "请输入要添加的关键词"
                
                # 执行单个文件的关键词添加
                return process_keyword_operation([file_path], keyword, position, "add")
        
            def remove_keyword(file_path, keyword):
                if not file_path or not keyword:
                    # 添加详细错误信息，区分路径和关键词
                    if not file_path and not keyword:
                        return "请提供文件路径和关键词"
                    elif not file_path:
                        return "请提供有效的文件路径"
                    else:
                        return "请输入要删除的关键词"
                
                # 执行单个文件的关键词删除
                return process_keyword_operation([file_path], keyword, None, "remove")
        
            def process_keyword_operation(file_paths, keyword, position, operation):
                """处理关键词操作，支持单个或多个文件"""
                if not file_paths:
                    return "请选择至少一个文件"
                
                results = []
                success_count = 0
                failed_count = 0
                
                for file_path in file_paths:
                    if not os.path.exists(file_path):
                        results.append(f"❌ 跳过: 文件不存在 - {file_path}")
                        failed_count += 1
                        continue
                    
                    try:
                        # 使用utf-8编码读取文件
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
        
                        # 添加调试信息：显示原始文件内容
                        print(f"调试信息: 原始文件内容 - {os.path.basename(file_path)}")
                        print("原始内容开始")
                        print(content)
                        print("原始内容结束")
        
                        # 执行操作
                        if operation == "add":
                            if position == "开头":
                                new_content = keyword + '\n' + content
                            elif position == "结尾":
                                new_content = content + '\n' + keyword
                            else:  # 随机位置
                                import random
                                lines = content.split('\n')
                                idx = random.randint(0, len(lines))
                                lines.insert(idx, keyword)
                                new_content = '\n'.join(lines)
                        elif operation == "remove":
                            new_content = content.replace(keyword, "")
        
                        # 添加调试信息：显示处理后文件内容
                        print(f"调试信息: 处理后文件内容 - {os.path.basename(file_path)}")
                        print("处理后内容开始")
                        print(new_content)
                        print("处理后内容结束")
        
                        # 使用utf-8编码写入文件
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
        
                        if operation == "add":
                            results.append(f"✅ 关键词 '{keyword}' 已添加到 {os.path.basename(file_path)}")
                        else:
                            results.append(f"✅ 关键词 '{keyword}' 已从 {os.path.basename(file_path)} 删除")
                        success_count += 1
                    except PermissionError:
                        error_msg = f"❌ 权限错误：无法访问文件 - {os.path.basename(file_path)}\n"
                        error_msg += "可能的解决方案：\n"
                        error_msg += "1. 确保文件未被其他程序占用\n"
                        error_msg += "2. 检查文件是否设置了只读属性\n"
                        error_msg += "3. 以管理员身份运行程序\n"
                        error_msg += "4. 检查文件所在文件夹的访问权限"
                        results.append(error_msg)
                        failed_count += 1
                    except Exception as e:
                        results.append(f"❌ 错误: {str(e)} - {os.path.basename(file_path)}")
                        failed_count += 1
                
                summary = f"操作完成: 成功 {success_count} 个文件, 失败 {failed_count} 个文件\n"
                return summary + "\n".join(results)
        
            def get_all_files(path):
                """获取指定路径下的所有txt文件"""
                if not path or not os.path.isdir(path):
                    return []
                return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]
        
            def process_keyword_operation_all(path, keyword, position, operation):
                """处理关键词操作，批量处理所有文件"""
                all_files = get_all_files(path)
                if not all_files:
                    return "请先加载文件列表"
                
                if operation == "add":
                    return process_keyword_operation(all_files, keyword, position, "add")
                else:
                    return process_keyword_operation(all_files, keyword, None, "remove")
        
            # 当文件夹路径发生变化时更新文件列表
            folder_path.change(update_folder_path, inputs=[folder_path], outputs=[folder_path, file_list, selected_file, file_path])
            
            # 当选择的文件发生变化时更新文件路径
            selected_file.change(update_selected_file, inputs=[selected_file, folder_path], outputs=[file_path])
            
            # 单个文件操作
            add_btn.click(add_keyword, inputs=[file_path, keyword_input, position], outputs=[result_output])
            remove_btn.click(remove_keyword, inputs=[file_path, keyword_input], outputs=[result_output])
            
            # 批量操作
            add_to_all_btn.click(
                lambda path, keyword, position: process_keyword_operation_all(path, keyword, position, "add"),
                inputs=[folder_path, keyword_input, position], 
                outputs=[result_output]
            )
            
            remove_from_all_btn.click(
                lambda path, keyword: process_keyword_operation_all(path, keyword, None, "remove"),
                inputs=[folder_path, keyword_input], 
                outputs=[result_output]
            )
            
            # 添加调试信息输出
            

            # 将关键组件保存到result中供外部调用
            result["folder_path"] = folder_path
            result["selected_file"] = selected_file
            result["file_list"] = file_list
            result["file_path"] = file_path
            result["keyword_input"] = keyword_input
            result["position"] = position
            result["add_btn"] = add_btn
            result["remove_btn"] = remove_btn
            result["add_to_all_btn"] = add_to_all_btn
            result["remove_from_all_btn"] = remove_from_all_btn
            result["result_output"] = result_output

    return result