# Agent 插件保护方案

本插件采用“入口明文，核心混淆”的保护方式。

## 为什么不加密整个目录

Stable Diffusion WebUI 必须能读取并导入 `scripts/agent.py`，否则插件不会加载。  
因此入口文件需要保留明文，核心逻辑文件再通过 PyArmor 混淆。

## 保护内容

建议混淆：

- `scripts/agent_config.py`
- `scripts/agent_tools.py`
- `scripts/agent_tools_registry.py`
- `scripts/agent_prompts.py`
- `scripts/canva_mcp.py`

保留明文：

- `scripts/agent.py`
- `agent_config.json`

注意：`agent_config.json` 只能保存空 Key 或默认配置，不要发布真实 API Key。

## 生成保护版

在插件目录执行：

```powershell
python -m pip install pyarmor
.\protect_agent_pyarmor.ps1 -Python python
```

如果使用 Forge 自带 Python：

```powershell
..\..\..\system\python\python.exe -m pip install pyarmor
.\protect_agent_pyarmor.ps1 -Python ..\..\..\system\python\python.exe
```

生成目录：

```text
dist/sd-webui-agent-protected
```

把这个目录作为发布版插件即可。

## 重要限制

这种方式能显著提高复制和二次改名成本，但不能做到绝对防逆向。  
只要代码能在用户机器上运行，高级逆向仍然有理论可能。
