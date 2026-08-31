<p align="center">
  <img src=".github/banner.png" alt="Banner"/>
</p>
# ✏️ Prompt All-in-One Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/eduardoabreu81/sd-webui-prompt-all-in-one-neo?style=flat-square)](https://github.com/eduardoabreu81/sd-webui-prompt-all-in-one-neo/stargazers)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

> ⚠️ **This is a fork.** All core functionality was created by [Physton](https://github.com/Physton) in the original [sd-webui-prompt-all-in-one](https://github.com/Physton/sd-webui-prompt-all-in-one). This Neo fork exists solely to maintain compatibility with Forge Neo and to apply community-reported UX fixes that were never merged upstream. If you are not running Forge Neo, **please use the original extension instead**.

---

## 📋 Table of Contents

- [What's New](#-whats-new)
- [Changelog](#-changelog)
- [Roadmap](#️-roadmap)
- [Features](#-features)
  - [Quality Presets](#-quality-presets-)
- [Demo](#-demo)
- [Installation](#-installation)
- [Language Support](#-language-support)
- [Translation APIs](#-translation-apis)
- [Credits](#-credits)

---

## 🆕 What's New

### v0.3.3 — Drag Fix

> Fixes a drag-and-drop regression where tags could leak into other prompts.

- **Fixed selection-box interference during drag** — dragging a tag no longer accidentally triggers the multi-select box, which previously corrupted the DOM and caused tags to appear in the wrong prompt area

### v0.3.2 — Stability Patch

> Small patch focused on a smoother BREAK / AND experience.

- **Better issue detection for BREAK / AND interactions** — the extension now handles this scenario more reliably
- **Move behavior fixed in visual mode** — BREAK / AND can be moved normally again
- **Consistent quick actions** — delete and common actions stay available in separator mode

### v0.3.1 — UX Patch

> Quality-of-life patch on top of v0.3.0.

- **Prompt Format toggle for BREAK / AND visual separator** — choose between classic keyword chips and visual full-width separator rendering
- **BREAK / AND drag restored** — separator mode no longer blocks dragging
- **BREAK / AND quick delete restored** — separator mode keeps quick removal behavior via `x`

---

## 📖 Changelog

### v0.3.3 — Drag Fix
> Fixes a drag-and-drop regression introduced after the BREAK / AND separator improvements.

- Guarded the selection-box logic (`dropMixin.js`) so it aborts while a Sortable drag is active
- Prevents DOM corruption that caused dragged tags to appear in other prompts or lists

### v0.3.2 — Stability Patch
> Focused fix release for BREAK / AND interaction flow.

- Improved handling when BREAK / AND interaction problems are detected
- Restored smooth move behavior in visual separator mode
- Kept quick actions consistent in separator mode

### v0.3.1 — UX Patch
> Follow-up fix for BREAK / AND visual separator behavior.

- Added a **Prompt Format** toggle to enable/disable visual separator rendering for `BREAK` / `AND`
- Fixed a regression where visual separators could block drag-and-drop reordering
- Fixed quick delete usability for `BREAK` / `AND` in visual separator mode

### v0.3.0 — UX Features
> Two of the most-requested improvements from the community.

- **BREAK / AND visual separator** — `BREAK` and `AND` tokens now always render as a full-width `— BREAK —` divider in the tag area, making attention boundaries instantly visible (#278)
- **Export Favorites** — one-click download of all four favorites groups as a single JSON file (#330)
- **Import Favorites** — restore your favorites from a previously exported file; useful when switching installs or sharing presets (#330)

### v0.2.0 — Quality Presets
> Per-checkpoint quality tag presets with CivitAI auto-detection.

- **Templates tab** — built-in quality tag sets for Pony, NoobAI, Illustrious, SDXL, SD 1.5, and Flux.1; each family independently toggleable with editable positive and negative overrides
- **Checkpoints tab** — identifies any installed checkpoint’s model family by querying CivitAI; per-row Detect and Scan All buttons; results cached so the API is never called twice for the same file
- **Custom tab** — define your own quality tags matched by checkpoint filename; mark as auto-insert to have them applied whenever you switch to that model
- **Auto-inject on model switch** — quality tags are prepended automatically when you change checkpoints mid-session, if the matched preset has auto-insert enabled
- **CivitAI API key** stored in **Forge Settings → Prompt All-in-One Neo** (not in browser storage)
- Duplicate-safe: existing tags are never doubled when injecting presets

### v0.1.2 — Bug Fixes
> Two long-standing glitches eliminated.

- Fixed a startup flash / double-render that occurred on page load (#328, #289)
- Fixed LoRA tags incorrectly blinking as “missing” during the initial load before the model list was ready (#269)

### v0.1.1 — UX Improvement Batch
> Three community-reported issues from the upstream repo fixed.

- Fixed a rare crash when the extra-networks list was modified during iteration (#250)
- Integer weights now always display with one decimal place (e.g. `1.0`) — prevents layout shifts in the input field (#341)
- Fixed residual space left before/after LoRA tokens when removing them (#294)

### v0.1.0 — Forge Neo Baseline
> Critical compatibility fixes to make the extension load and run on Forge Neo.

- Removed dependency on a module that no longer exists in Forge Neo (`modules.sd_hijack`)
- Token counter updated to use the API Forge Neo exposes
- Safe fallback for a startup argument that was removed from Forge Neo

---

## 🗺️ Roadmap

### v0.3.3 — Drag Fix *(complete)*
- Fixed selection-box interference during Sortable drag ✅

### v0.3.2 — Stability Patch *(complete)*
- Better issue detection for BREAK / AND interactions ✅
- BREAK / AND move behavior fixed in visual mode ✅
- Quick actions preserved in separator mode ✅

### v0.1.0 — Forge Neo Baseline *(complete)*
- Full compatibility with Forge Neo ✅

### v0.1.1 — UX Improvement Batch *(complete)*
- Extra-networks crash fix ✅
- Consistent weight decimal format ✅
- Clean LoRA token removal ✅

### v0.1.2 — Bug Fixes *(complete)*
- Startup flash / double-render fix ✅
- LoRA tag blinking fix on load ✅

### v0.2.0 — Quality Presets *(complete)*
- Quality tag templates per model family (Pony, NoobAI, Illustrious, SDXL, SD 1.5, Flux.1) ✅
- CivitAI checkpoint detection with SHA-256 ✅
- Custom presets matched by filename with auto-insert ✅
- Auto-inject on model switch ✅
- CivitAI API key in Forge Settings ✅

### v0.3.0 — UX Features *(complete)*
- BREAK / AND visual separator in the tag area ✅
- Export / import favorites list ✅

### v0.3.1 — UX Patch *(complete)*
- Prompt Format toggle for BREAK / AND visual separator ✅
- BREAK / AND drag and quick delete behavior preserved in separator mode ✅

### v0.4.0 — Translation Security *(planned)*
- Secure storage for translation API credentials

### v1.0.0 — First Stable Release *(planned)*
- All known regressions resolved
- Full Forge Neo compatibility guarantee

### vNext — Codebase Refactor *(planned)*
- Rename internal identifiers from the original physton naming to `paio-neo`

---

## 🎯 Features

> ⭐ = added or fixed in this Neo fork · everything else is original work by [Physton](https://github.com/Physton)

### ✏️ Prompt Editing

- **Intuitive tag interface** — displays prompt tokens as individual styled chips with bilingual (source ↔ translated) comparison
- **Drag to reorder** — rearrange tags by dragging without retyping
- **One-click weight adjustment** — increase or decrease tag weight with `(` `)` brackets; configurable step size
- **Consistent weight format** — integer weights always display with one decimal place (e.g. `1.0`), preventing input-field layout shifts ⭐
- **NovelAI symbol mode** — switch between `()` and `{}` weight notation
- **Disable/enable tags** without deleting them
- **One-click delete** per tag or via batch box-select
- **BREAK / AND keyword support** — special tokens render correctly in the tag list

### 🔤 Translation

- **Auto-translate** — translates prompt/negative prompt automatically as you type
- **Batch translation** — translate all tags at once with one click
- **Dozens of translation services** — Google, Baidu, DeepL, Microsoft, OpenAI, Alibaba, Tencent, Yandex, and many more
- **API key optional** — most free services work without registration
- **Offline translation** — mBART-50 model supported for air-gapped environments
- **Translation history** — per-tag translated value stored and shown inline

### 🗂️ History & Favorites

- **Automatic history** — every prompt change is recorded
- **Favorites** — bookmark individual tags or entire prompts; one-click restore
- **Batch favorite** — box-select multiple tags and favorite them all at once

### 🔍 Extra Networks

- **LoRA / LyCORIS / Textual Inversion detection** — recognized tokens are highlighted with distinct colors
- **Existence check** — missing LoRA/embedding names are flagged visually
- **Popup info panel** — hover a LoRA tag to see its metadata, preview image, and trained keywords
- **No residual space after LoRA tokens** ⭐
- **Fixed extra-networks crash on concurrent load** ⭐

### 🤖 ChatGPT Integration

- **Generate prompts with ChatGPT** — describe your scene in plain language and get a prompt back
- **Configurable model and API key** — works with any OpenAI-compatible endpoint

### 🎯 Quality Presets ⭐

- **Templates** — built-in quality tag sets per model family (Pony, NoobAI, Illustrious, SDXL, SD 1.5, Flux.1); each family independently toggleable; editable positive + negative overrides per family
- **Checkpoint scanner** — identify any installed checkpoint's model family by querying CivitAI with its SHA-256 hash; results cached locally
- **Custom presets** — define your own quality tags matched by filename substring; mark as **auto-insert** to have them injected whenever you switch to a matching model
- **Auto-inject on model switch** — when a checkpoint change is detected mid-session and auto-insert is enabled, quality tags are prepended to the active tab automatically
- **CivitAI API key** stored in **Forge Settings → Prompt All-in-One Neo** (not in browser storage)

### ⚙️ Format & Settings

- **Prompt format options** — comma spacing, trailing comma removal, LoRA separator behavior, newline handling
- **Blacklist** — tags that are automatically filtered out
- **Hotkeys** — configurable keyboard shortcuts per action
- **Custom themes** — override CSS via built-in extension system
- **Token counter** — live token count and max-length indicator using Forge Neo's native `get_prompt_lengths_on_ui` API ⭐

---

## 🎬 Demo

*All demos below are from the original [physton/sd-webui-prompt-all-in-one](https://github.com/Physton/sd-webui-prompt-all-in-one) — full credit to [Physton](https://github.com/Physton).*

- **Switch language**

  ![](https://s1.imagehub.cc/images/2023/06/06/demo.switch_language.gif)

- **Translation API settings**

  ![](https://s1.imagehub.cc/images/2023/06/06/demo.translate_setting.gif)

- **Show/Hide panel**

  ![](https://s1.imagehub.cc/images/2023/06/06/demo.fold.gif)

- **Automatic translation**

  ![](https://s1.imagehub.cc/images/2023/06/06/demo.auto_translate.gif)

- **One-click translation**

  ![](https://s1.imagehub.cc/images/2023/06/06/demo.translate.gif)

- **Elegant input**

  ![](https://s1.imagehub.cc/images/2023/06/06/demo.elegant_input.gif)

- **Quick weight adjustment**

  ![](https://s1.imagehub.cc/images/2023/06/06/demo.quick_adjust.gif)

- **Favorite and History**

  ![](https://s1.imagehub.cc/images/2023/06/06/demo.history_favorite.gif)

- **Use ChatGPT to generate prompts**

  ![](https://s1.imagehub.cc/images/2023/06/06/demo.chatgpt.gif)

- **LoRA / LyCORIS / Textual Inversion highlighting**

  ![](https://s1.imagehub.cc/images/2023/06/06/demo.keyword_detection.gif)

- **Prompt format options**

  ![](https://s1.imagehub.cc/images/2023/06/06/demo.prompt_format.gif)

- **Batch operations**

  ![](https://s1.imagehub.cc/images/2023/06/06/demo.batch_operation.gif)

- **Custom theme / extension styles**

  ![](https://s1.imagehub.cc/images/2023/06/06/demo.custom_theme.gif)

- **One-click keyword addition**

  ![](https://s1.imagehub.cc/images/2023/08/15/demo.group_tags.gif)

---

## 📦 Installation

### For Forge Neo

1. Open Forge Neo WebUI
2. Navigate to **Extensions** → **Install from URL**
3. Paste: `https://github.com/eduardoabreu81/sd-webui-prompt-all-in-one-neo`
4. Click **Install** and reload WebUI

### Requirements

- ✅ [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
- ✅ Python 3.10+

> ⚠️ **Not using Forge Neo?** Use the [original physton extension](https://github.com/Physton/sd-webui-prompt-all-in-one) instead. This fork will not work correctly on Automatic1111 or Forge Classic.

---

## 🌐 Language Support

The UI itself is available in 12 languages:

<details>
<summary>UI Supported languages</summary>

`简体中文` `繁體中文` `English` `Русский` `日本語` `한국어` `Français` `Deutsch` `Español` `Português` `Italiano` `العربية`

</details>

Translation supports 100+ languages:

<details>
<summary>Translation Supported languages</summary>

`简体中文 (中国)` `繁體中文 (中國香港)` `繁体中文 (中國台灣)` `English (US)` `Afrikaans (South Africa)` `Shqip (Shqipëria)` `አማርኛ (ኢትዮጵያ)` `العربية (السعودية)` `Հայերեն (Հայաստան)` `অসমীয়া (ভাৰত)` `Azərbaycan dili (Latın, Azərbaycan)` `বাংলা (বাংলাদেশ)` `Башҡорт (Россия)` `Euskara (Espainia)` `Bosanski (Latinski, Bosna i Hercegovina)` `Български (България)` `Català (Espanya)` `Hrvatski (Hrvatska)` `Čeština (Česká republika)` `Dansk (Danmark)` `Nederlands (Nederland)` `Eesti (Eesti)` `Filipino (Pilipinas)` `Suomi (Suomi)` `Français (France)` `Français (Canada)` `Galego (España)` `ქართული (საქართველო)` `Deutsch (Deutschland)` `Ελληνικά (Ελλάδα)` `ગુજરાતી (ભારત)` `עברית (ישראל)` `हिन्दी (भारत)` `Magyar (Magyarország)` `Bahasa Indonesia (Indonesia)` `Gaeilge (Éire)` `Italiano (Italia)` `日本語 (日本)` `ಕನ್ನಡ (ಭಾರತ)` `Қазақ (Қазақстан)` `ភាសាខ្មែរ (កម្ពុជា)` `한국어 (대한민국)` `Кыргызча (Кыргызстан)` `ລາວ (ລາວ)` `Latviešu (Latvija)` `Lietuvių (Lietuva)` `Македонски (Северна Македонија)` `Bahasa Melayu (Latin, Malaysia)` `മലയാളം (ഇന്ത്യ)` `Malti (Malta)` `Māori (Aotearoa)` `मराठी (भारत)` `Монгол (Кирилл, Монгол улс)` `မြန်မာ (မြန်မာ)` `नेपाली (नेपाल)` `Norsk bokmål (Norge)` `فارسی (ایران)` `Polski (Polska)` `Português (Brasil)` `Português (Portugal)` `Română (România)` `Русский (Россия)` `Српски (ћирилица, Србија)` `Slovenčina (Slovensko)` `Slovenščina (Slovenija)` `Soomaali (Soomaaliya)` `Español (España)` `Kiswahili (Kenya)` `Svenska (Sverige)` `தமிழ் (இந்தியா)` `తెలుగు (భారత)` `ไทย (ไทย)` `Türkçe (Türkiye)` `Українська (Україна)` `اردو (پاکستان)` `O'zbekcha (Lotin, O'zbekiston)` `Tiếng Việt (Việt Nam)` `Cymraeg (Y Deyrnas Unedig)` `isiZulu (iNingizimu Afrika)` and more…

</details>

---

## 🌐 Translation APIs

### No API Key Required

Free to use, but may be rate-limited or unstable. If translation fails, switch to another service.

### API Key Required

Most have a free tier — register and obtain a key:

| Service | Free Tier |
|---|---|
| DeepL | ✅ |
| Microsoft Translator | ✅ |
| OpenAI | ❌ |
| Baidu | ✅ |
| Alibaba | ✅ |
| Tencent | ✅ |
| Yandex | ✅ |
| Caiyun | ✅ |
| Niutrans | ✅ |
| iFlytek | ✅ |
| Volcengine | ✅ |
| Amazon Translate | ✅ |

### Offline

- **mBART-50** — downloads a local model on first use (~1.5 GB); works without internet access after download

---

## 📄 Credits

### Original Project — all core functionality

> This fork would not exist without the extensive work of [Physton](https://github.com/Physton). Every feature in this extension was originally designed, built, and maintained by him. Please consider giving a ⭐ to the original repository.

**[sd-webui-prompt-all-in-one](https://github.com/Physton/sd-webui-prompt-all-in-one)** by **[Physton](https://github.com/Physton)**

- Intuitive tag-based prompt interface
- Auto-translate and batch translation with 100+ languages
- History, favorites, blacklist, hotkeys
- ChatGPT integration
- LoRA / LyCORIS / embedding detection and metadata popup
- Custom themes and extension system
- Full multilingual UI (12 languages)

### Neo Fork — Forge Neo compatibility + upstream fixes

**[sd-webui-prompt-all-in-one-neo](https://github.com/eduardoabreu81/sd-webui-prompt-all-in-one-neo)** by **[Eduardo Abreu](https://github.com/eduardoabreu81)**

- Full compatibility with Forge Neo
- Quality Presets with CivitAI model detection
- BREAK / AND visual separators, Favorites export / import
- Various UX fixes from upstream community reports

### Special Thanks

- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by [Haoming02](https://github.com/Haoming02)
- All contributors who reported and diagnosed upstream issues

---

## 📜 License

MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ for the Stable Diffusion community

**[Report Bug](https://github.com/eduardoabreu81/sd-webui-prompt-all-in-one-neo/issues)** • **[Request Feature](https://github.com/eduardoabreu81/sd-webui-prompt-all-in-one-neo/issues)** • **[Discussions](https://github.com/eduardoabreu81/sd-webui-prompt-all-in-one-neo/discussions)** • **[☕ Ko-fi](https://ko-fi.com/eduardoabreu81)**

Original extension by **[Physton](https://github.com/Physton/sd-webui-prompt-all-in-one)** — please ⭐ the original repo if you find this useful.

</div>
