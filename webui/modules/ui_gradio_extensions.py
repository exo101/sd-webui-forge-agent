import os
from urllib.parse import quote

import gradio as gr

from modules import localization, scripts, shared, util
from modules.paths import data_path, script_path


def webpath(fn):
    # URLs must use forward slashes.  On Windows, relpath() returns
    # backslashes; browsers normalize those differently and extension
    # scripts can fail to load while Gradio remains stuck on "Loading".
    path = util.truncate_path(fn).replace("\\", "/")
    return f"gradio_api/file={quote(path, safe='/')}?{os.path.getmtime(fn)}"


def javascript_html():
    # Ensure localization is in `window` before scripts
    head = f'<script type="text/javascript">{localization.localization_js(shared.opts.localization)}</script>\n'

    script_js = os.path.join(script_path, "script.js")
    head += f'<script type="text/javascript" src="{webpath(script_js)}"></script>\n'

    for script in scripts.list_scripts("javascript", ".js"):
        head += f'<script type="text/javascript" src="{webpath(script.path)}"></script>\n'

    for script in scripts.list_scripts("javascript", ".mjs"):
        head += f'<script type="module" src="{webpath(script.path)}"></script>\n'

    if shared.cmd_opts.theme:
        head += f'<script type="text/javascript">set_theme("{shared.cmd_opts.theme}");</script>\n'

    return head


def css_html():
    head = ""

    def stylesheet(fn):
        return f'<link rel="stylesheet" property="stylesheet" href="{webpath(fn)}">'

    for cssfile in scripts.list_files_with_name("style.css"):
        head += stylesheet(cssfile)

    user_css = os.path.join(data_path, "user.css")
    if os.path.exists(user_css):
        head += stylesheet(user_css)

    from modules.shared_gradio_themes import resolve_var

    light = resolve_var("background_fill_primary")
    dark = resolve_var("background_fill_primary_dark")
    head += f"<style>html {{ background-color: {light}; }} @media (prefers-color-scheme: dark) {{ html {{background-color:  {dark}; }} }}</style>"

    return head


def reload_javascript():
    js = javascript_html()
    css = css_html()

    def template_response(*args, **kwargs):
        res = shared.GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b"</head>", f'{js}<meta name="referrer" content="no-referrer"/></head>'.encode("utf8"))
        res.body = res.body.replace(b"</body>", f"{css}</body>".encode("utf8"))
        # 页面语言声明为中文：阻止 Edge/Chrome 的"翻译此页"功能介入。
        # 浏览器翻译插件重写文本节点会与 webui 的 DOM 监听回调互相触发，
        # 形成无限变更循环把页面主线程冻死（表现为"此页面没有响应"）
        res.body = res.body.replace(b'lang="en"', b'lang="zh-CN"')
        res.init_headers()
        # 页面 HTML 不缓存：避免浏览器缓存旧页面（引用旧 JS 文件），
        # 导致修改 JS 后普通刷新仍加载旧代码（本地服务，无性能影响）
        res.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return res

    gr.routes.templates.TemplateResponse = template_response


if not hasattr(shared, "GradioTemplateResponseOriginal"):
    shared.GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse
