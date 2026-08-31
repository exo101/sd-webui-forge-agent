<div align="center">
  <img src=".github/logo.png?v=2" alt="CivitAI Browser Neo"/>
</div>

# 🎨 CivitAI Browser Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Gradio](https://img.shields.io/badge/Gradio-4.40.0-orange)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Browse, download, and manage your CivitAI models directly inside Forge Neo — with auto-organization, disk usage dashboard, creator management, and support for all modern architectures (FLUX, Wan, Qwen, Pony, Illustrious, and more).

---

## 📋 Table of Contents

- [What's New](#-whats-new)
- [Changelog](#-changelog)
- [Roadmap](#️-roadmap)
- [Features](#-features)
- [Installation](#-installation)
- [Auto-Organization System](#-auto-organization-system)
- [Dashboard & Statistics](#-dashboard--statistics)
- [Supported Model Types](#-supported-model-types)
- [Credits](#-credits)

---

## 🆕 What's New

### v0.9.0 — Major Update: CivitAI Domain Support, Update Mode Isolation & Download Resilience

- **Full support for the new CivitAI domain split** — CivitAI now separates SFW content (`civitai.com`) from the complete catalog (`civitai.red`). The extension adapts automatically so nothing breaks. Paste any CivitAI link from either domain and it opens instantly.
- **New "SFW only" setting** — a simple checkbox in Settings lets you restrict all links and API calls to `civitai.com` if you prefer. Off by default, so the full catalog stays accessible without extra steps.
- **Update Mode filter isolation** — Browser-tab filters are ignored when Update Mode is active, preventing "No updates match the current filters" false negatives. `pressRefresh()` and page-slider triggers no longer pull Browser filters into the Update Mode view.
- **SHA256 silent-update detection** — if a downloaded file fails hash verification, the extension re-queries the CivitAI API for the version's current SHA256. If the author updated the file silently, the new hash is accepted and metadata is updated instead of failing.
- **Defensive ambiguity handling** — searching by SHA256 that returns multiple candidates no longer crashes with `KeyError`; the code safely falls back to an error message.
- **Exact search restricted to Model name** — CivitAI API does not support quoted search for Tag or User name. Exact search now only wraps terms in quotes when "Model name" is selected.
- **Batch download resilience** — internal loop eliminates gaps between queued items, timeout increased to 30s with automatic retry (up to 3×), SHA256 buffer increased to 8MB.
- **Update list sorted by mtime** — outdated models are now listed with most recently modified first, making it easier to prioritize updates.

---

## 📖 Changelog

### v0.9.0 — CivitAI Domain Support, Update Mode Isolation & Download Resilience
- Added centralized domain helper (`get_civitai_domain()`) to replace all hardcoded `civitai.com` URLs across the extension.
- Added `civitai_sfw_only` checkbox setting (default: off → `civitai.red`) to toggle between domains.
- Fixed search-box direct-link parser to recognize both `civitai.com` and `civitai.red` URLs.
- Updated all API calls, model page links, uploader profile links, `Referer` headers, and JSON sidecar `modelPageURL` fields to use the configured domain.
- `initial_model_page`: verify `gl.update_mode` before resetting state; ignore Browser-tab filters when Update Mode is active.
- `download_create_thread`: on SHA256 mismatch, re-query `/api/v1/model-versions/{version_id}` to detect silent file updates by the author.
- `create_model_item` / `selected_to_queue`: propagate `version_id` through the download queue for post-download API recheck.
- `resolve_ambiguity`: defensively sync `model_sha256` and `model_filename` with the chosen candidate.
- Hardened `initial_model_page` and `update_model_info` against malformed `gl.json_data` (e.g. SHA256 ambiguity dicts without `items`/`metadata`).
- Exact search restricted to Model name only — CivitAI API does not support quoted search for Tag or User name.
- Batch download internal loop eliminates Gradio event-queue gaps between items.
- Timeout increased to 30s with automatic retry (up to 3×) for network failures.
- SHA256 verification buffer increased from 1MB to 8MB.
- Update list sorted by file modification time (most recent first).
- Fix re-trigger loop on `queue_trigger` preventing duplicate download executions.

### v0.8.3 — Safer Delete Flow for Installed/Outdated Models
- Browser version dropdown now defaults to an installed version whenever one exists, even when updates are available.
- This preserves delete action availability for models that are installed but outdated.
- Added a delete failsafe for card quick delete: if multiple installed versions exist, quick delete is canceled and the user is instructed to pick the exact installed version in the Browser panel before deleting.
- Installed-model Browser loading now keeps unmatched files visible as local-only entries while still using API data for matched models.

### v0.8.2 — Checkpoint SHA256 Cache Sync
- Added automatic SHA256 cache sync for checkpoints right after successful download completion.
- Added a manual `Sync checkpoint SHA256 cache` button in Update Models to reconcile local checkpoints against Forge cache.
- Added local checkpoint hash registry (`lib/models/checkpoint_hashes.json`) to track synced entries and clean stale records for deleted files.

### v0.8.1 — Trigger Word Bugfixes & Resilience
- Fixed an issue where the local trigger word fallback process ignored API groups and flattened words into single lines.
- Fixed an issue where "Update model info & tags" didn't safely persist incoming `trainedWords` groups natively to the local cache.
- Added exponential backoff retry mechanism to API calls returning temporary 50x server errors, saving "Update models" loops from failing silently on affected files.
- Moved trigger word row buttons (📋 / ➕) to the left side of the text in the preview panel.

### v0.8.0 — Trigger Word Consolidation
- Consolidated trigger words from `.safetensors` metadata, local `.json` `activation text`, and API `trainedWords`
- Added case-insensitive deduplication while preserving original order
- Model info now uses local consolidated trigger words first, with API fallback when local cache is unavailable

### v0.7.4 — Wan I2V/T2V Differentiation
- Wan card badges now distinguish `I2V`, `T2V`, and `TI2V` subtypes (API already returns specific `baseModel` values)
- New setting `civitai_neo_wan_subfolder_by_type` (OFF by default): splits Wan downloads into `Wan/I2V/`, `Wan/T2V/`, `Wan/TI2V/` subfolders
- Fixed multi-level subfolder "already organized" check (was using `os.path.basename`, broke for `Wan/I2V` paths)
- Fixed Flux.2 Klein 4B/9B and Flux.2 D showing `F1` badge — now correctly shows `F2`

### v0.7.3 — Per-group Trigger Word Rows
- Each trigger word group gets its own row with individual copy and add-to-prompt buttons
- LORA tag row (`<lora:filename:1>`) shown as first entry in purple/monospace
- Clipboard copy with ✓ visual feedback (1.5s)
- "Add all to prompt" button when multiple groups exist

### v0.7.2 — Bug Fixes
- Fixed wildcard base-model subfolder being applied in GUI even when `wildcard_organize_by_base` is OFF (`civitai_api.py` path calculation was missing the wildcard guard)
- Fixed delete-by-SHA256 silently failing — `json_base` path not joined with `root`, making `os.path.exists` search in CWD instead of the model directory

### v0.7.1 — Wildcard Download Improvements
- Own subfolder per wildcard download (sd-dynamic-prompts compatible)
- Flat zip extraction — no double-nesting when the zip has internal folders
- Skip preview/gallery images for Wildcards
- New settings: `wildcard_own_folder` (ON by default), `wildcard_organize_by_base` (OFF by default)

### v0.6.3 — Download Reliability
- **File integrity check** — every download is verified against CivitAI's expected hash after completing; corrupted or incomplete files are detected and removed automatically
- **Faster batch queueing** — adding many models to the queue is now instant; metadata is fetched only when each download actually starts, not at queue time
- **Safer cancel** — cancelling a download or clearing the queue is now more reliable and won't cause hangs
- **Aria2 auto-reconnect** — if the download engine crashes during a session, it automatically restarts and resumes the current download
- **Skipped model feedback** — if a model can't be queued during batch selection, a visible error message appears in the Download Manager

### v0.6.2 — Queue Persistence
- **Download queue survives disconnects** — the queue is saved server-side; if your session ends (e.g. RunPod timeout), a restore banner appears next time you open the UI
- **One-click restore** — re-queue all interrupted downloads with a single button click

### v0.6.1 — Companion Files Banner
- **Companion files guidance** — a warning banner appears when downloading models that require additional files (e.g. Wan 2.2, FLUX, Qwen), with specific instructions per architecture
- **Background download fix** — downloads no longer stall when the browser tab is in the background

### v0.6.0 — Creator Management
- **Favorite creators** — mark creators as favorites; their cards get a gold glow and ⭐ badge
- **Ban creators** — hide cards from creators you don't want to see; one-click toggle
- **Persistent lists** — favorites and bans are saved to disk and restored across sessions
- **Mutually exclusive** — favoriting a creator removes them from the ban list and vice versa

### v0.5.0 — Dashboard Export & Rankings
- **Export dashboard** — download your collection stats as CSV or JSON
- **Top 10 largest files** — see which individual models take the most space
- **Top 10 categories by file count**
- **Orphan detection** — optional scan to find model files with missing or incomplete metadata

### v0.4.5 — Removed Model Handling
- **Graceful 404 handling** — when a model has been deleted from CivitAI, the extension shows the locally cached info instead of an error, with a clear warning banner

### v0.4.4 — Model Info Polish
- **LoRA activation syntax** — "Add to prompt" now inserts the correct `<lora:filename:1>` syntax automatically
- **SHA256 in model info** — hash shown in the version details; click to select for easy copy
- **Live progress bars** — Validate and Fix operations now show real-time progress

### v0.4.3 — Organization Validator
- **Validate organization** — read-only scan that shows which models are correctly placed, misplaced, or missing metadata
- **Fix misplaced files** — move flagged models to the correct folders in one click; backup created automatically
- **Instant undo** — revert the fix immediately without scrolling to the Undo section

### v0.4.2
- "Add to prompt" button now works in the model info overlay (CivitAI icon in txt2img/img2img)
- Various crash fixes

### v0.4.1
- Shift+click on meta field buttons to append to prompt instead of replacing
- Deleted models go to the OS recycle bin by default
- Search settings now persist correctly across restarts

### v0.4.0 — Update Intelligence
- Dashboard banner showing outdated model counts after scanning
- Batch update — select multiple outdated models and download all at once
- Version comparison by model family (not just version name)
- Retention policy on update: keep, move to trash, or replace

### v0.3.x
- Card color legend bar above the browser grid
- DNS retry on transient network failures
- Dashboard UX improvements (hide empty categories, scan summary)

### v0.1.0 — Neo Baseline
- Full Forge Neo / Gradio 4 compatibility
- Auto-organization system with backup and rollback
- Extended architecture support

---

## 🗺️ Roadmap

### v0.7.0 — Forge Neo Compatibility *(complete)* ✅

### v0.7.1 — Wildcard Download Improvements *(complete)* ✅

### v0.7.2 — Bug Fixes *(complete)* ✅

### v0.7.3 — Per-group Trigger Word Rows *(complete)* ✅

### v0.7.4 — Wan I2V/T2V Differentiation *(complete)* ✅

### v0.8.0 — Trigger Word Consolidation *(complete)* ✅

### v0.8.1 — Trigger Word Bugfixes & Resilience *(complete)* ✅

### v0.8.2 — Checkpoint SHA256 Cache Sync *(complete)* ✅

### v0.8.3 — Safer Delete Flow for Installed/Outdated Models *(complete)* ✅

### v0.9.0 — CivitAI Domain Support, Update Mode Isolation & Download Resilience *(complete)* ✅

### v0.10.0 — Advanced Curation *(planned)*
- Saved search presets
- Favorites in creator/user search
- Additional browser quality-of-life improvements
- **Organization by Tag — Phase 1**: save CivitAI tags to `.json` sidecar; editable user-tags field in model panel for manual assignment
- **Organization by Tag — Phase 2**: in Manage tab, pick "anchor" tags → models with that tag sort into `<type>/<tag>/` subfolders (independent of base-model organization)

### v1.0.0 — First Stable Release *(planned)*
- All known issues resolved
- Full Forge Neo compatibility guarantee

---

## 🎯 Features

> ⭐ = exclusive to Neo

### 🔍 Browse & Search

- Browse CivitAI directly inside the WebUI — no tab switching
- Search by model name, tag, or username
- Filter by content type: Checkpoint, LORA, VAE, ControlNet, Upscaler, TextualInversion, Wildcards, Workflows, and more
- Filter by base model: SD 1.x, SDXL, Pony, Illustrious, FLUX, Wan, Qwen, NoobAI, Lumina, and more — list auto-updated from CivitAI at startup ⭐
- Sort by: Highest Rated, Most Downloaded, Newest, Most Liked, Most Discussed
- Filter by time period: Day, Week, Month, Year, All Time
- NSFW toggle, liked-only filter, hide installed models, hide banned creators
- Exact search mode
- Search settings persist across restarts ⭐

### 📥 Download

- Download any model, version, and file variant directly
- High-speed multi-connection downloads via Aria2 (optional, on by default)
- Download queue — multiple downloads run in sequence without blocking the UI
- Queue persistence — survives session disconnects with one-click restore ⭐
- Cancel individually or clear the entire queue
- Folder automatically set based on content type ⭐
- Custom sub-folders per download
- API key support for early access and private models
- Proxy support for restricted regions

### 🔄 Model Updates ⭐

- Orange border on cards with a newer version available
- Batch update — select multiple outdated models and download all at once
- Version comparison by model family (not just version name)
- Retention policy on update: keep, move to trash, or replace
- Dashboard shows outdated model counts after scanning

### 🗂️ Auto-Organization ⭐

- New downloads automatically sorted into subfolders by base model (SDXL/, Pony/, FLUX/, etc.)
- Organize your existing collection in one click
- Validate organization — read-only check showing correct / misplaced / no-metadata per file ⭐
- Fix misplaced files in one click — automatic backup created first ⭐
- One-click rollback (keeps last 5 backups)
- Custom folder mapping in Settings
- Associated files (`.json`, `.png`, `.txt`) always move with the model

### 🖼️ Model Info & Preview

- Model info panel with name, version, base model, type, tags, permissions, and description
- Sample images with "Send to txt2img" — fills prompt, negative, sampler, steps, CFG
- Individual meta field buttons — send just one field; Shift+click to append ⭐
- "➕ Add to prompt" in the model overlay — appends trigger words directly; auto-inserts LoRA syntax ⭐
- SHA256 hash shown in version info — click to select ⭐
- Video preview on hover for cards with video samples ⭐
- Save model info and images locally

### 📊 Dashboard ⭐

- Disk usage by category and architecture
- Pie chart with percentage breakdown
- Top 10 largest files and categories
- Orphan file detection (optional)
- Export to CSV or JSON
- Update summary after scanning

### 🃏 Model Cards

- Color-coded borders: aquamarine = installed, orange = outdated, gold = early access / favorite creator
- Color legend bar always visible above the grid ⭐
- NSFW, Early Access (💎), and type badges
- Configurable tile size
- Quick delete from the card
- Multi-select checkboxes for batch download ⭐
- Favorite (⭐) and ban (🚫) creator directly from the card ⭐

### 🔒 Safety

- Deleted models go to the OS recycle bin by default (configurable)
- Filename sanitization — removes illegal characters automatically
- Filename length capped to prevent filesystem errors

---

## 📦 Installation

1. Open Forge Neo WebUI
2. Go to **Extensions** → **Install from URL**
3. Paste: `https://github.com/eduardoabreu81/sd-civitai-browser-neo`
4. Click **Install** and reload the WebUI

> ⚠️ This extension requires **Forge Neo**. For Forge Classic or Automatic1111, use the [anxety-solo fork](https://github.com/anxety-solo/sd-civitai-browser-plus).

---

## 📁 Auto-Organization System

The organization system sorts your models into subfolders based on their base model type, using the metadata saved alongside each file.

**Before:**
```
models/Lora/
├── model1.safetensors  (SDXL)
├── model2.safetensors  (Pony)
└── model3.safetensors  (FLUX)
```

**After:**
```
models/Lora/
├── SDXL/model1.safetensors
├── Pony/model2.safetensors
└── FLUX/model3.safetensors
```

### Auto-organize new downloads
Enable **"Auto-organize downloads"** in Settings → Model Organization.

### Organize existing models
Go to **Update Models** tab → select types → click **"📁 Organize models into subfolders"**.

### Safety
- Automatic backup before any operation
- One-click undo
- Conflict detection (skips files that already exist at destination)

---

## 📊 Dashboard & Statistics

Go to the **Dashboard** tab, select the content types you want to analyze, and click **"📊 Generate Dashboard"**.

You'll see:
- Total file count and disk usage
- Breakdown by category and architecture (Checkpoints and LORAs split by Pony, SDXL, FLUX, etc.)
- Visual progress bars and pie chart
- Top 10 largest files and categories
- Optional orphan file detection

Results can be exported as CSV or JSON.

---

## 🎨 Supported Model Types

| Architecture | Notes |
|---|---|
| SD 1.x / SD 2.x | Classic Stable Diffusion |
| SDXL | Base SDXL and derivatives |
| Pony | Pony V6 and variants |
| Illustrious | Illustrious XL |
| NoobAI | NoobAI (Illustrious-based) |
| FLUX | Dev, Krea, Kontext, Klein |
| Wan | Wan 2.2 — text/image to video |
| Qwen | Qwen-Image, Qwen-Image-Edit |
| Z-Image | Z-Image, Z-Image Turbo |
| Lumina | Lumina-Image 2.0 |
| Anima | Anima |
| Chroma | Chroma1-HD |
| Cascade | Stable Cascade |
| SVD | Stable Video Diffusion |
| Hunyuan | Hunyuan |
| Other | Catch-all; fully configurable |

Custom categories can be defined in **Settings → Model Organization** using a simple JSON pattern list.

---

## 📄 Credits

- **[sd-civitai-browser](https://github.com/Vetchems/sd-civitai-browser)** by Vetchems — original project
- **[sd-civitai-browser-plus](https://github.com/BlafKing/sd-civitai-browser-plus)** by BlafKing — foundation for this fork
- **[sd-civitai-browser-plus](https://github.com/anxety-solo/sd-civitai-browser-plus)** by anxety-solo — UI redesign and quality improvements
- **[sd-webui-civbrowser](https://github.com/SignalFlagZ/sd-webui-civbrowser)** by SignalFlagZ — creator management inspiration
- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02

---

## 📜 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">

Made with ❤️ for the Stable Diffusion community

**[Report Bug](https://github.com/eduardoabreu81/sd-civitai-browser-neo/issues)** • **[Request Feature](https://github.com/eduardoabreu81/sd-civitai-browser-neo/issues)** • **[Discussions](https://github.com/eduardoabreu81/sd-civitai-browser-neo/discussions)** • **[☕ Ko-fi](https://ko-fi.com/eduardoabreu81)**

</div>
