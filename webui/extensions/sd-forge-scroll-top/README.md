# Forge Neo 一键回顶悬浮球

一个纯前端扩展，不修改 Forge Neo 核心文件。

## 功能
- 点击悬浮球：平滑回到 Forge 页面顶部
- 兼容浏览器页面滚动与 Gradio/Forge Neo 内部滚动容器
- 可直接拖动悬浮球到任意位置
- 位置会自动保存到浏览器 localStorage
- 右键悬浮球：恢复默认右下角位置
- 不需要 Python 依赖

## 安装
将整个 `sd-forge-scroll-top` 文件夹放入：

`webui/extensions/`

最终应为：

`webui/extensions/sd-forge-scroll-top/javascript/scroll_top.js`
`webui/extensions/sd-forge-scroll-top/style.css`

然后完整重启 Forge Neo。

## 卸载/停用
在 Forge Neo 的扩展管理中取消勾选 `sd-forge-scroll-top`，或者删除该文件夹。

## 说明
本插件只操作页面滚动，不接触模型、采样器、提示词或生成流程。
