# Prompt All-in-One Neo + Hy-MT2 merge

Base: `sd-webui-prompt-all-in-one-neo`.

Changes:
- Preserves Neo Quality Presets and Forge Neo compatibility changes.
- Ports Tencent Hunyuan Hy-MT2 translation and lazy initialization from the Hy-MT2 fork.
- Fixes Hy-MT2 model switching and target-language selection.
- Fixes the Forge Neo startup 404 race: the frontend automatically retries the backend config endpoint instead of requiring a manual browser refresh.

Install:
1. Disable/remove the separate `sd-webui-prompt-all-in-one-Hy-MT2` extension (otherwise both extensions register the same `/physton_prompt/*` routes).
2. Replace your current `sd-webui-prompt-all-in-one-neo` folder with this folder.
3. Restart Forge Neo once.

Hy-MT2 remains lazy-loaded and is only loaded when initialized/used.
