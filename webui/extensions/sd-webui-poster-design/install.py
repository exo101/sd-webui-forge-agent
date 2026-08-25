"""海报设计插件 - 安装依赖检查"""

import os
import launch

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

if os.path.exists(req_file):
    launch.run_pip(f"install -r \"{req_file}\"", "海报设计插件依赖")
