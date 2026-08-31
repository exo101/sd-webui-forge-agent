/**
 * Sidebar Navigation for SD WebUI
 * Apple-style sidebar - the ONLY way to navigate tabs.
 * All Gradio tab panels are hidden by default.
 * Clicking a sidebar item shows the corresponding panel.
 */

// Wrap entire file in IIFE so the global guard's `return` is valid
(function() {
    'use strict';

// Source guard: Gradio auto-loads javascript/*.js via <script src> with mtime.
// Those can be browser-cached. Only execute when loaded via fetch+eval
// (document.currentScript is null in that case), which always gets latest version.
if (document.currentScript && document.currentScript.src && document.currentScript.src.indexOf('v=') === -1) {
    console.log('[SD] Skipping Gradio auto-loaded sidebar.js (cached script tag)');
    return;
}

// Global guard: prevent double execution when both Gradio auto-load
// and ui.py dynamic injection load this file
// NOTE: Do NOT return here - always let the latest version run to apply new features.
// injectSidebar and moveControlsToSidebar have their own guards to prevent duplicates.
if (window.__sdSidebarLoaded) {
    // Re-inject latest theme CSS even if already loaded (handles version updates)
    fetch('/gradio_api/file=theme.css?v=81&t=' + Date.now(), { cache: 'no-store' })
        .then(function(r) { return r.text(); })
        .then(function(css) {
            var old = document.getElementById('sd-theme-style');
            if (old) old.remove();
            var style = document.createElement('style');
            style.id = 'sd-theme-style';
            style.textContent = css;
            document.head.appendChild(style);
        })
        .catch(function(e) { console.error('[SD] theme.css refresh failed:', e); });
    // Fall through to execute latest version (don't return!)
}
window.__sdSidebarLoaded = true;

// ============================================================
//  Clean up legacy v18-v21 duplicate "控制辅助" section
//  (Old cached sidebar.js created an extra sd-sidebar-controls-section;
//   remove it while preserving any real Gradio controls inside it)
// ============================================================
(function cleanupLegacyControls() {
    // 1. Remove old legacy controls sections
    var legacySections = document.querySelectorAll('.sd-sidebar-controls-section');
    legacySections.forEach(function(sec) { sec.remove(); });
    var orphan = document.getElementById('sd-controls-body');
    if (orphan && !orphan.closest('.sd-sidebar-section')) orphan.remove();

    // 2. Rescue seed/styles elements that old cached sidebar.js moved into .sd-control-group wrappers
    var popupIds = ['txt2img_seed_row', 'img2img_seed_row', 'txt2img_styles_row', 'img2img_styles_row'];
    popupIds.forEach(function(id) {
        var el = document.getElementById(id);
        if (!el) return;
        var wrapper = el.closest('.sd-control-group');
        if (wrapper) {
            // Remove the wrapper, keep the element in place where it is
            var parent = wrapper.parentNode;
            if (parent) parent.insertBefore(el, wrapper);
            wrapper.remove();
            delete el.dataset.sdMoved;
            delete el.dataset.sdUpgraded;
            delete el.dataset.sdCollapseInit;
        }
    });

    // 3. Remove any orphan .sd-control-group wrappers left in sidebar
    document.querySelectorAll('.sd-sidebar .sd-control-group').forEach(function(g) { g.remove(); });

    // 4. Strip sd-popup-target / sd-popup-visible classes from any elements.
    //    The seed popup feature was removed — seed rows now live in their native
    //    Gradio positions. Old cached sidebar.js may have added these classes,
    //    which would keep seed rows hidden via CSS. Remove them to restore visibility.
    document.querySelectorAll('.sd-popup-target, .sd-popup-visible').forEach(function(el) {
        el.classList.remove('sd-popup-target');
        el.classList.remove('sd-popup-visible');
    });
    // Remove any orphan popup close buttons
    document.querySelectorAll('.sd-popup-close').forEach(function(btn) { btn.remove(); });
    document.body.classList.remove('sd-popup-seed-open', 'sd-popup-styles-open');
})();

// ============================================================
//  Inject Theme CSS dynamically (avoids Gradio URL rewriting)
// ============================================================
(function() {
    'use strict';
    var STYLE_ID = 'sd-theme-style';

    fetch('/gradio_api/file=theme.css?v=81&t=' + Date.now(), { cache: 'no-store' })
        .then(function(r) { return r.text(); })
        .then(function(css) {
            var old = document.getElementById(STYLE_ID);
            if (old) old.remove();
            var style = document.createElement('style');
            style.id = STYLE_ID;
            style.textContent = css;
            document.head.appendChild(style);
            console.log('[SD] theme.css injected, length:', css.length);
        })
        .catch(function(e) {
            console.error('[SD] Failed to load theme.css:', e);
        });
})();

(function() {
    'use strict';

    // ============================================================
    //  Sidebar Configuration (User-defined modules)
    // ============================================================

    // Custom sidebar sections.
    // Only "控制辅助" remains — it manages embedded accordions on the
    // txt2img/img2img pages (ControlNet, ADetailer, 高清修复, etc.),
    // which are NOT separate plugin tabs. All external plugin tabs are
    // now listed in the unified "插件" section with inline eye-toggles.
    var sidebarModules = [
        {
            name: '控制辅助',
            icon: '🎯',
            items: [
                { label: '通配符', tabId: 'txt2img', accordionId: 'sddp-dynamic-prompting' },
                { label: 'ControlNet', tabId: 'txt2img', accordionId: 'controlnet' },
                { label: '场景编辑器', tabId: 'txt2img', accordionId: 'RP_maint2i', subTabLabel: '🎯 区域提示' },
                { label: 'ADetailer 面部修复', tabId: 'txt2img', accordionId: 'script_txt2img_adetailer_ad_main_accordion' },
                { label: '高清修复', tabId: 'txt2img', accordionId: 'txt2img_hr' },
                { label: '多图拼接参考', tabId: 'txt2img', accordionId: 'label:多图参考' },
                { label: '脚本', tabId: 'txt2img' },
            ]
        }
    ];

    // ============================================================
    //  Chinese display name mapping for tab IDs
    // ============================================================

    var tabChineseNames = {
        'txt2img': '文生图',
        'img2img': '图生图',
        'extras': '后期处理',
        'pnginfo': 'PNG信息',
        'settings': '设置',
        'extensions': '扩展',
        'modelmerger': '模型合并',
        'tutorial_center': '教程中心',
        'forge_h3_studio': 'MiniMax H3 工作台',
        'aesthetic_enhancement_tab': '视觉分析',
        'multimodal_media_tab': '多媒体处理',
        'Segmentation_Tab': '智能抠图',
        'trellis2_3d_generator': 'TRELLIS 图生3D',
        'tagger': '标签器',
        'sddp-wildcard-manager': '通配符管理',
        'sd_forge_image_stitch': '多图拼接参考'
    };

    function getChineseName(tabId, fallbackLabel) {
        if (tabChineseNames[tabId]) return tabChineseNames[tabId];
        // If the label is mostly English/ID-like, return Chinese name or id
        if (fallbackLabel && /[\u4e00-\u9fa5]/.test(fallbackLabel)) return fallbackLabel;
        return tabChineseNames[tabId] || fallbackLabel || tabId;
    }

    // ============================================================
    //  Collect all Gradio tabs dynamically
    // ============================================================

    var allTabPanels = []; // {id, label, elem, buttonText}

    // Native Forge tabs — matched by both tab ID AND button text to be
    // robust against Gradio version changes in DOM ID format.
    var builtinTabIds = [
        'txt2img', 'img2img', 'extras', 'pnginfo', 'settings', 'extensions',
        'modelmerger', 'sd_forge_image_stitch'
    ];
    var builtinTabTexts = [
        // Chinese (current localization)
        '文生图', '图生图', '后期处理', 'PNG信息', '设置', '扩展',
        '模型合并', '多图拼接参考',
        // English fallbacks (for other localizations)
        'txt2img', 'img2img', 'Extras', 'PNG Info', 'Settings', 'Extensions',
        'Model Merger', 'Multi-image Reference'
    ];

    function collectTabs() {
        allTabPanels = [];
        // Only collect TOP-LEVEL main tab buttons inside #tabs (the main
        // Gradio tab bar). Must scope to the MAIN tablist — querySelectorAll
        // at any depth would also match nested sub-tablists inside txt2img/
        // img2img panels (ControlNet units, LoRA tabs, etc.).
        var tabBar = document.getElementById('tabs');
        if (!tabBar) return;
        var seenPanelIds = {};

        // Gradio 6.x: main tablist is inside #tabs > .tab-wrapper > .tab-container
        // Gradio 5.x: main tablist is a direct child of #tabs
        // We try multiple selectors in order of specificity.
        var tablist = null;
        // 1. Gradio 6.x structure: #tabs > .tab-wrapper > .tab-container[role=tablist]
        var wrappers = tabBar.querySelectorAll(':scope > .tab-wrapper, :scope > div.tab-wrapper');
        for (var w = 0; w < wrappers.length; w++) {
            var tl = wrappers[w].querySelector(':scope > .tab-container[role="tablist"]');
            if (tl) { tablist = tl; break; }
        }
        // 2. Gradio 5.x structure: direct child of #tabs
        if (!tablist) {
            tablist = tabBar.querySelector(':scope > [role="tablist"]');
        }
        // 3. Fallback: first tablist anywhere in #tabs (but verify it's not nested)
        if (!tablist) {
            var allTablists = tabBar.querySelectorAll('[role="tablist"]');
            for (var t = 0; t < allTablists.length; t++) {
                // A nested tablist is inside a [role=tabpanel]
                if (!allTablists[t].closest('[role="tabpanel"]')) {
                    tablist = allTablists[t];
                    break;
                }
            }
        }

        var tabButtons = tablist ? tablist.querySelectorAll(':scope > button[role="tab"]') : [];
        // Fallback: direct button children of #tabs (Gradio 4 style)
        if (tabButtons.length === 0) {
            tabButtons = tabBar.querySelectorAll(':scope > button[role="tab"]');
        }
        tabButtons.forEach(function(btn) {
            var panelId = btn.getAttribute('aria-controls') || '';
            if (!panelId || seenPanelIds[panelId]) return;
            var panel = document.getElementById(panelId);
            if (!panel) return;
            // Skip panels that are nested inside another tabpanel (sub-tabs)
            if (panel.closest('[role="tabpanel"]') && panel.closest('[role="tabpanel"]') !== panel) return;
            seenPanelIds[panelId] = true;
            var id = panelId.replace(/^tabs?_/, '').replace(/^tab_/, '');
            var buttonText = btn.textContent.trim();
            var label = getChineseName(id, buttonText || id);
            allTabPanels.push({ id: id, label: label, elem: panel, buttonText: buttonText });
        });
    }

    function isBuiltinTab(panel) {
        if (builtinTabIds.indexOf(panel.id) !== -1) return true;
        if (panel.buttonText && builtinTabTexts.indexOf(panel.buttonText) !== -1) return true;
        if (panel.label && builtinTabTexts.indexOf(panel.label) !== -1) return true;
        return false;
    }

    // ============================================================
    //  Tab visibility management (localStorage persistence)
    // ============================================================

    function isTabHidden(tabId) {
        // Built-in/native tabs are ALWAYS visible (never hidden)
        if (builtinTabIds.indexOf(tabId) !== -1) return false;
        // External plugins: default to HIDDEN from top bar.
        // Only shown if user explicitly opted in via eye-toggle.
        var visiblePlugins = getVisiblePlugins();
        return visiblePlugins.indexOf(tabId) === -1;
    }

    function getVisiblePlugins() {
        try {
            var raw = localStorage.getItem('sd-visible-plugins');
            if (!raw) return [];
            var arr = JSON.parse(raw);
            return Array.isArray(arr) ? arr : [];
        } catch (e) {
            return [];
        }
    }

    function setVisiblePlugins(ids) {
        try {
            localStorage.setItem('sd-visible-plugins', JSON.stringify(ids));
        } catch (e) {
            // localStorage unavailable; ignore
        }
    }

    function setTabHidden(tabId, hidden) {
        // For built-in tabs, do nothing (they're always visible)
        if (builtinTabIds.indexOf(tabId) !== -1) return;

        // Manage the "explicitly visible" set for external plugins.
        // hidden=true  → remove from visible set (hide from top bar)
        // hidden=false → add to visible set (show in top bar)
        var visiblePlugins = getVisiblePlugins();
        var vidx = visiblePlugins.indexOf(tabId);
        if (!hidden && vidx === -1) {
            visiblePlugins.push(tabId);
        } else if (hidden && vidx !== -1) {
            visiblePlugins.splice(vidx, 1);
        }
        setVisiblePlugins(visiblePlugins);
        applyTabVisibility(tabId);
    }

    function applyTabVisibility(tabId) {
        // Find real panel ID from allTabPanels, fallback to "tab_" + id
        var panelId = null;
        for (var i = 0; i < allTabPanels.length; i++) {
            if (allTabPanels[i].id === tabId) {
                panelId = allTabPanels[i].elem.id;
                break;
            }
        }
        if (!panelId) panelId = 'tab_' + tabId;
        var btn = document.querySelector('button[aria-controls="' + panelId + '"]');
        if (!btn) {
            // Fallback: partial match
            btn = document.querySelector('button[aria-controls*="' + tabId + '"]');
        }
        if (!btn) return;
        if (isTabHidden(tabId)) {
            btn.classList.add('sd-tab-hidden');
        } else {
            btn.classList.remove('sd-tab-hidden');
        }
    }

    function applyAllTabVisibility() {
        // Safety: ensure allTabPanels is populated before iterating.
        if (allTabPanels.length === 0) collectTabs();
        // Use allTabPanels (collected with robust ID extraction) to find
        // buttons by their real panel ID — avoids hardcoding "tab_" prefix.
        allTabPanels.forEach(function(panel) {
            var btn = document.querySelector('button[aria-controls="' + panel.elem.id + '"]');
            if (!btn) {
                // Fallback: try matching by partial panel ID
                btn = document.querySelector('button[aria-controls*="' + panel.id + '"]');
            }
            if (btn) applyTabVisibility(panel.id);
        });
    }

    // ============================================================
    //  Item availability check (hide items whose extension/tab isn't installed)
    // ============================================================

    function itemAvailable(item) {
        // Popup items (seed/styles): available if target elements exist in DOM
        if (item.popup) {
            if (item.popup === 'seed') {
                return document.getElementById('txt2img_seed_row') !== null ||
                       document.getElementById('img2img_seed_row') !== null;
            }
            if (item.popup === 'styles') {
                return document.getElementById('txt2img_styles_row') !== null ||
                       document.getElementById('img2img_styles_row') !== null;
            }
            return false;
        }

        // The target tab must exist in the DOM
        var tabExists = allTabPanels.some(function(t) { return t.id === item.tabId; });
        if (!tabExists) return false;

        // Container-based items (sampler, seed, styles): at least one container must exist
        if (item.containerIds && item.containerIds.length > 0) {
            return item.containerIds.some(function(id) { return document.getElementById(id); });
        }

        // Accordion-based items: the accordion element must exist
        if (item.accordionId) {
            if (item.accordionId.indexOf('label:') === 0) {
                var labelText = item.accordionId.substring(6);
                var spans = document.querySelectorAll('.input-accordion .label-wrap span');
                for (var i = 0; i < spans.length; i++) {
                    if (spans[i].textContent.trim() === labelText) return true;
                }
                return false;
            }
            return document.getElementById(item.accordionId) !== null;
        }

        // Sub-tab items: a matching button must exist inside the panel
        if (item.subTabLabel) {
            var panel = document.getElementById('tab_' + item.tabId);
            if (!panel) return false;
            var buttons = panel.querySelectorAll('button');
            for (var i = 0; i < buttons.length; i++) {
                if (buttons[i].textContent.trim() === item.subTabLabel) return true;
            }
            return false;
        }

        return true;
    }

    // ============================================================
    //  Panel visibility control
    // ============================================================

    function showOnlyPanel(panelId) {
        allTabPanels.forEach(function(panel) {
            var isActive = panel.id === panelId;
            panel.elem.classList.toggle('sd-panel-active', isActive);
            panel.elem.classList.toggle('sd-panel-hidden', !isActive);
        });
    }

    // ============================================================
    //  Embedded plugin accordion visibility control
    //  These accordions are embedded in txt2img/img2img pages.
    //  Hidden by default; shown when clicked from the sidebar.
    // ============================================================

    var embeddedAccordionIds = [
        'txt2img_hr',
        'sddp-dynamic-prompting',
        'script_txt2img_adetailer_ad_main_accordion',
        'script_img2img_adetailer_ad_main_accordion',
        // RP_maint2i / RP_maini2i removed from hidden list — the user expects
        // the "场景编辑器" accordion (containing the camera angle selector) to
        // be directly visible on the txt2img/img2img pages.
        'controlnet',
    ];

    var scriptContainerIds = [
        'txt2img_script_container',
        'img2img_script_container',
    ];

    function hideEmbeddedAccordions() {
        // Only hide accordions explicitly listed by ID.
        // Do NOT use document.querySelectorAll('.input-accordion') — it can
        // match accordions inside gallery groups or other containers, and
        // el.closest('.group') would hide the ENTIRE parent group including
        // the gallery, making all images invisible.
        embeddedAccordionIds.forEach(function(id) {
            var el = document.getElementById(id);
            if (!el) return;
            var container = el.closest('.group') || el.parentElement;
            if (container) container.style.display = 'none';
            else el.style.display = 'none';
        });
        scriptContainerIds.forEach(function(id) {
            var el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });
    }

    function showEmbeddedAccordion(id) {
        var el = document.getElementById(id);
        if (!el && id && id.indexOf('label:') === 0) {
            var labelText = id.substring(6);
            var allAccordions = document.querySelectorAll('.input-accordion');
            for (var i = 0; i < allAccordions.length; i++) {
                var labelEl = allAccordions[i].querySelector('.label-wrap span');
                if (labelEl && labelEl.textContent.trim() === labelText) {
                    el = allAccordions[i];
                    break;
                }
            }
        }
        if (!el) return;

        // 找到实际需要控制可见性的容器
        var scriptContainer = el.closest('#txt2img_script_container, #img2img_script_container');
        var container = el.closest('.group') || el.parentElement;
        var target = container || el;

        // 独立切换当前 accordion：已显示则隐藏，已隐藏则显示
        var isVisible = target.style.display !== 'none';
        if (isVisible) {
            target.style.display = 'none';
            return;
        }

        // 显示当前 accordion（不影响其他已打开的 accordion）
        if (scriptContainer) scriptContainer.style.display = '';
        target.style.display = '';

        // 展开 accordion 内容（如果还没展开）
        var labelWrap = el.querySelector('.label-wrap');
        if (labelWrap) {
            // 检查 accordion 是否已经展开（Gradio 5 的 accordion 有 open 属性/aria-expanded）
            var isOpen = el.classList.contains('open') || el.hasAttribute('open') || el.getAttribute('aria-expanded') === 'true';
            if (!isOpen) {
                labelWrap.dispatchEvent(new MouseEvent('click', {
                    bubbles: true, cancelable: true, view: window
                }));
            }
        }
    }

    // ============================================================
    //  Switch to a tab (sidebar manages only non-main tabs)
    // ============================================================

    // Tabs managed by Gradio's native top bar (no sidebar panel manipulation)
    var nativeTabIds = ['txt2img', 'img2img', 'extras', 'pnginfo', 'tutorial_center', 'settings', 'extensions'];

    function switchTab(tabId, showAccordionId, subTabLabel, containerIds) {
        if (!tabId) return;

        // Find the actual panel in allTabPanels to get its real DOM ID.
        // This is robust against Gradio version changes in ID format
        // (e.g. tab_<id> vs tabs_<id> vs hashed IDs).
        var panelId = null;
        for (var i = 0; i < allTabPanels.length; i++) {
            if (allTabPanels[i].id === tabId) {
                panelId = allTabPanels[i].elem.id;
                break;
            }
        }
        if (!panelId) panelId = 'tab_' + tabId; // fallback

        // Just click the native Gradio tab button — let Gradio handle
        // panel visibility natively. Do NOT manually add/remove
        // sd-panel-hidden classes, because that can break nested content
        // inside Settings/Extensions pages (accordion blocks, sub-tabs,
        // extension management UI, etc.).
        var tabButton = document.querySelector('button[aria-controls="' + panelId + '"]');
        if (tabButton) {
            tabButton.click();
        } else {
            // Fallback: try matching by button text or partial aria-controls
            var fallbackBtn = document.querySelector('button[aria-controls*="' + tabId + '"]');
            if (fallbackBtn) fallbackBtn.click();
        }

        // Only manage embedded accordions on txt2img/img2img pages
        if (showAccordionId) {
            showEmbeddedAccordion(showAccordionId);
        } else if (tabId === 'txt2img' || tabId === 'img2img') {
            // For "脚本" item (no accordionId), show the script container
            var sc = document.getElementById(tabId + '_script_container');
            if (sc) sc.style.display = '';
        }

        if (subTabLabel) {
            switchSubTab(tabId, subTabLabel);
        }
    }

    function switchSubTab(panelId, label) {
        var panel = document.getElementById('tab_' + panelId);
        if (!panel) return;
        // Find the tab button inside the panel whose text matches the label
        var buttons = panel.querySelectorAll('button');
        buttons.forEach(function(btn) {
            if (btn.textContent.trim() === label) {
                btn.click();
            }
        });
    }

    // ============================================================
//  Inject Sidebar HTML
// ============================================================

// Build sidebar inner HTML from current allTabPanels.
// Extracted so rebuildSidebarHTML() can reuse it.
function buildSidebarHTML() {
    var html = '';
    html += '<div class="sd-sidebar-brand">';
    html += '  <div class="sd-sidebar-logo">SD</div>';
    html += '  <div class="sd-sidebar-title">';
    html += '    <h1>Stable Diffusion</h1>';
    html += '    <p>Forge NEO</p>';
    html += '  </div>';
    html += '</div>';
    html += '<nav class="sd-sidebar-nav">';

    // Custom modules (currently only "控制辅助" for embedded accordions)
    sidebarModules.forEach(function(mod, mi) {
        var availableItems = mod.items.filter(function(item) { return itemAvailable(item); });
        if (availableItems.length === 0) return;

        html += '<div class="sd-sidebar-section">';
        html += '  <div class="sd-sidebar-section-header" data-section="' + mi + '">';
        html += '    <span class="sd-sidebar-section-icon">' + mod.icon + '</span>';
        html += '    <span class="sd-sidebar-section-label">' + mod.name + '</span>';
        html += '    <span class="sd-sidebar-chevron">▾</span>';
        html += '  </div>';
        html += '  <div class="sd-sidebar-section-body">';
        availableItems.forEach(function(item) {
            var popupAttr = item.popup ? ' data-popup="' + item.popup + '"' : '';
            html += '    <div class="sd-sidebar-item" data-tab="' + (item.tabId || '') + '"' + popupAttr + ' data-accordion="' + (item.accordionId || '') + '" data-subtab="' + (item.subTabLabel || '') + '" data-containers="' + (item.containerIds ? item.containerIds.join(',') : '') + '">';
            html += '      <span class="sd-sidebar-item-label">' + item.label + '</span>';
            html += '    </div>';
        });
        html += '  </div>';
        html += '</div>';
    });

    // Unified "插件" section: ALL external plugin tabs with eye-toggles
    var externalPlugins = allTabPanels.filter(function(panel) {
        return !isBuiltinTab(panel) && panel.id !== 'sd_webui_agent';
    });
    if (externalPlugins.length > 0) {
        html += '<div class="sd-sidebar-section">';
        html += '  <div class="sd-sidebar-section-header" data-section="plugins">';
        html += '    <span class="sd-sidebar-section-icon">🧩</span>';
        html += '    <span class="sd-sidebar-section-label">插件</span>';
        html += '    <span class="sd-sidebar-chevron">▾</span>';
        html += '  </div>';
        html += '  <div class="sd-sidebar-section-body">';
        externalPlugins.forEach(function(panel) {
            var isHidden = isTabHidden(panel.id);
            var eyeIcon = isHidden ? '🚫' : '👁';
            html += '    <div class="sd-sidebar-item has-eye-toggle" data-tab="' + panel.id + '">';
            html += '      <span class="sd-sidebar-item-label">' + panel.label + '</span>';
            html += '      <span class="sd-sidebar-eye-toggle' + (isHidden ? ' hidden' : '') + '" data-tab-id="' + panel.id + '" title="' + (isHidden ? '显示标签页' : '隐藏标签页') + '">' + eyeIcon + '</span>';
            html += '    </div>';
        });
        html += '  </div>';
        html += '</div>';
    }

    // Dedicated "绘梦智能体助手" — standalone nav item below plugins
    var agentPanel = null;
    for (var i = 0; i < allTabPanels.length; i++) {
        if (allTabPanels[i].id === 'sd_webui_agent') {
            agentPanel = allTabPanels[i];
            break;
        }
    }
    if (agentPanel) {
        html += '<div class="sd-sidebar-item" data-tab="sd_webui_agent" style="margin:4px 8px;">';
        html += '  <span class="sd-sidebar-item-label">' + agentPanel.label + '</span>';
        html += '</div>';
    }

    html += '</nav>';
    html += '<div class="sd-sidebar-footer"></div>';
    return html;
}

function injectSidebar() {
    if (document.getElementById('sd-sidebar')) return;

    collectTabs();

    var sidebar = document.createElement('aside');
    sidebar.id = 'sd-sidebar';
    sidebar.className = 'sd-sidebar';
    sidebar.innerHTML = buildSidebarHTML();

    var toggle = document.createElement('button');
    toggle.id = 'sd-sidebar-toggle';
    toggle.className = 'sd-sidebar-toggle';
    toggle.innerHTML = '☰';
    toggle.setAttribute('title', '切换侧边栏');

    document.body.appendChild(sidebar);
    document.body.appendChild(toggle);
    document.body.classList.add('sd-sidebar-active');
}

// Rebuild sidebar HTML when new plugin tabs appear after initial render.
// Handles Gradio 6.x async Svelte rendering where tab buttons appear late.
function rebuildSidebarHTML() {
    var sidebar = document.getElementById('sd-sidebar');
    if (!sidebar) return;
    collectTabs();
    if (allTabPanels.length === 0) return;
    sidebar.innerHTML = buildSidebarHTML();
    // Re-bind events for new elements
    setupSidebarEvents();
    // Re-apply visibility preferences
    document.querySelectorAll('[id^="tab_"]').forEach(function(panel) {
        panel.classList.remove('sd-panel-hidden');
    });
    applyAllTabVisibility();
    // Re-add theme & home buttons to the freshly rebuilt footer.
    // rebuildSidebarHTML() wipes the innerHTML which clears any buttons
    // previously appended by the theme manager.
    if (window.SDThemeManager && window.SDThemeManager.rebuildFooter) {
        window.SDThemeManager.rebuildFooter();
    }
}

    // ============================================================
    //  Setup events
    // ============================================================

    function setupSidebarEvents() {
        var sidebar = document.getElementById('sd-sidebar');
        if (!sidebar) return;

        // Guard toggle button against duplicate event binding
        var toggle = document.getElementById('sd-sidebar-toggle');
        if (toggle && !toggle.__sdToggleBound) {
            toggle.__sdToggleBound = true;
            toggle.addEventListener('click', function() {
                document.body.classList.toggle('sd-sidebar-collapsed');
            });
        }

        // Section header expand/collapse
        sidebar.querySelectorAll('.sd-sidebar-section-header').forEach(function(header) {
            header.addEventListener('click', function(e) {
                e.stopPropagation();
                this.closest('.sd-sidebar-section').classList.toggle('expanded');
            });
        });

        // Eye-toggle: click to show/hide the tab button in top bar.
        // Stops propagation so it doesn't trigger the parent item's tab switch.
        sidebar.querySelectorAll('.sd-sidebar-eye-toggle').forEach(function(eye) {
            eye.addEventListener('click', function(e) {
                e.stopPropagation();
                var tabId = this.dataset.tabId;
                if (!tabId) return;
                var nowHidden = !isTabHidden(tabId);
                setTabHidden(tabId, nowHidden);
                // Update this eye icon
                this.classList.toggle('hidden', nowHidden);
                this.textContent = nowHidden ? '🚫' : '👁';
                this.title = nowHidden ? '显示标签页' : '隐藏标签页';
                // Also update any duplicate eye toggles for the same tabId
                sidebar.querySelectorAll('.sd-sidebar-eye-toggle[data-tab-id="' + tabId + '"]').forEach(function(other) {
                    if (other !== eye) {
                        other.classList.toggle('hidden', nowHidden);
                        other.textContent = nowHidden ? '🚫' : '👁';
                        other.title = nowHidden ? '显示标签页' : '隐藏标签页';
                    }
                });
            });
        });

        // Sidebar item click -> switch tab OR toggle popup
        sidebar.querySelectorAll('.sd-sidebar-item, .sd-sidebar-footer-item').forEach(function(item) {
            item.addEventListener('click', function(e) {
                e.stopPropagation();

                // Popup items (seed/styles): toggle floating panel, don't switch tabs
                var popup = this.dataset.popup;
                if (popup) {
                    togglePopup(popup, this);
                    return;
                }

                var tab = this.dataset.tab;
                var accordionId = this.dataset.accordion;
                var subTabLabel = this.dataset.subtab;
                var containerIdsStr = this.dataset.containers;
                var containerIds = containerIdsStr ? containerIdsStr.split(',').filter(function(s) { return s; }) : null;
                if (!tab) return;
                // Remove active from all items
                sidebar.querySelectorAll('.sd-sidebar-item, .sd-sidebar-footer-item').forEach(function(el) {
                    el.classList.remove('active');
                });
                this.classList.add('active');
                // Auto-expand parent section
                var section = this.closest('.sd-sidebar-section');
                if (section) section.classList.add('expanded');
                switchTab(tab, accordionId, subTabLabel, containerIds);
            });
        });
    }

    // ============================================================
    //  Observe Gradio tab changes
    // ============================================================

    function observeTabChanges() {
        var observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'attributes' && mutation.attributeName === 'aria-selected') {
                    var target = mutation.target;
                    if (target.getAttribute('aria-selected') === 'true') {
                        var tabContainer = target.closest('[id^="tab_"]');
                        if (tabContainer) {
                            var tabId = tabContainer.id.replace('tab_', '');

                            // Update sidebar highlight to match the first item with this tabId
                            var sidebar = document.getElementById('sd-sidebar');
                            if (sidebar) {
                                sidebar.querySelectorAll('.sd-sidebar-item, .sd-sidebar-footer-item').forEach(function(el) {
                                    el.classList.remove('active');
                                });
                                var firstMatch = sidebar.querySelector('.sd-sidebar-item[data-tab="' + tabId + '"]');
                                if (firstMatch) firstMatch.classList.add('active');
                            }
                        }
                    }
                }
            });
        });

        // Observe existing tabs
        document.querySelectorAll('[id^="tab_"]').forEach(function(container) {
            var btn = container.querySelector('button');
            if (btn) {
                observer.observe(btn, { attributes: true, attributeFilter: ['aria-selected'] });
            }
        });

        // Observe new tabs and Svelte re-renders
        // When Gradio re-renders tab buttons (Svelte), new DOM elements
        // are created without the sd-tab-hidden class. We must re-apply
        // visibility to keep user's toggle preferences in effect.
        //
        // IMPORTANT: This observer must NOT fire on changes WE make ourselves,
        // otherwise applyAllTabVisibility() -> DOM change -> observer ->
        // applyAllTabVisibility() creates an infinite loop causing flickering.
        var sdApplyingVisibility = false; // guard flag
        var sdVisibilityTimer = null;     // debounce timer

        var bodyObserver = new MutationObserver(function(mutations) {
            // Skip if we are currently modifying the DOM ourselves
            if (sdApplyingVisibility) return;

            var newTabs = document.querySelectorAll('[id^="tab_"]:not([data-sidebar-observed])');
            var hasNewTabs = newTabs.length > 0;
            newTabs.forEach(function(container) {
                container.setAttribute('data-sidebar-observed', 'true');
                var btn = container.querySelector('button');
                if (btn) {
                    observer.observe(btn, { attributes: true, attributeFilter: ['aria-selected'] });
                }
            });

            // Only re-apply visibility when genuinely new tabs appeared,
            // NOT on every mutation (prevents infinite loop).
            // Debounce so rapid successive mutations only trigger once.
            if (hasNewTabs) {
                if (sdVisibilityTimer) clearTimeout(sdVisibilityTimer);
                sdVisibilityTimer = setTimeout(function() {
                    sdApplyingVisibility = true;
                    try {
                        applyAllTabVisibility();
                    } finally {
                        sdApplyingVisibility = false;
                    }
                }, 150);
            }

            // Gradio 6.x: if sidebar exists but plugin section is empty and
            // new tab buttons have appeared, rebuild sidebar HTML.
            var sidebar = document.getElementById('sd-sidebar');
            if (sidebar && !sidebar.__sdRebuildScheduled && hasNewTabs) {
                var pluginSection = sidebar.querySelector('[data-section="plugins"]');
                var hasPluginItems = pluginSection && pluginSection.querySelectorAll('.sd-sidebar-item').length > 0;
                if (!hasPluginItems) {
                    var tabsEl = document.getElementById('tabs');
                    if (tabsEl) {
                        var btns = tabsEl.querySelectorAll('button[role="tab"]');
                        if (btns.length > 2) {
                            sidebar.__sdRebuildScheduled = true;
                            setTimeout(function() {
                                sdApplyingVisibility = true;
                                try {
                                    rebuildSidebarHTML();
                                } finally {
                                    sdApplyingVisibility = false;
                                    sidebar.__sdRebuildScheduled = false;
                                }
                            }, 300);
                        }
                    }
                }
            }
        });
        // Observe only #tabs container, not entire body, to avoid sidebar
        // internal changes (clicks, rebuilds) triggering this observer.
        var tabsRoot = document.getElementById('tabs');
        if (tabsRoot) {
            bodyObserver.observe(tabsRoot, { childList: true, subtree: true });
        } else {
            // Fallback: observe body but rely on sdApplyingVisibility guard
            bodyObserver.observe(document.body, { childList: true, subtree: true });
        }
    }

    // ============================================================
    //  Fix overflow tabs (PNG Info, Settings, Extensions, Extras)
    //  Gradio 5 collapses tabs that don't fit into .overflow-menu /
    //  .overflow-dropdown. Our theme CSS hides .overflow-menu entirely,
    //  which also hides its child tab buttons (children cannot override
    //  a parent's display:none via CSS). Solution: move the tab buttons
    //  out of the overflow area into the main .tab-container as siblings
    //  of txt2img/img2img buttons, then hide the overflow trigger.
    // ============================================================

    function fixOverflowTabs() {
        var tabs = document.getElementById('tabs');
        if (!tabs) return false;
        var tabWrapper = tabs.querySelector('.tab-wrapper');
        if (!tabWrapper) return false;

        // Find the VISIBLE tab-container (Gradio has a .visually-hidden one for measurement)
        var visibleContainer = null;
        var containers = tabWrapper.querySelectorAll(':scope > div.tab-container');
        containers.forEach(function(c) {
            if (!c.classList.contains('visually-hidden') && !visibleContainer) {
                visibleContainer = c;
            }
        });
        if (!visibleContainer) return false;

        // Whitelist of essential tabs that must be visible in the main bar.
        // Other plugin tabs stay hidden per existing theme CSS L1392.
        var whitelist = ['tab_txt2img', 'tab_img2img', 'tab_extras', 'tab_pnginfo', 'tab_tutorial_center', 'tab_settings', 'tab_extensions'];

        // Collect whitelisted buttons from visible container + overflow dropdown + overflow menu
        var overflowMenu = tabWrapper.querySelector(':scope > .overflow-menu');
        var overflowDropdown = tabWrapper.querySelector(':scope > .overflow-dropdown');
        var sources = [visibleContainer];
        if (overflowDropdown) sources.push(overflowDropdown);
        if (overflowMenu) sources.push(overflowMenu);

        var essentialBtns = {};
        sources.forEach(function(src) {
            if (!src) return;
            src.querySelectorAll('button[aria-controls^="tab_"]').forEach(function(btn) {
                var controls = btn.getAttribute('aria-controls') || '';
                if (whitelist.indexOf(controls) !== -1 && !essentialBtns[controls]) {
                    essentialBtns[controls] = btn;
                }
            });
        });

        // Force essential buttons visible (override Gradio's responsive display:none)
        function forceVisible(btn) {
            btn.style.setProperty('display', 'inline-flex', 'important');
            btn.style.setProperty('visibility', 'visible', 'important');
            btn.style.setProperty('opacity', '1', 'important');
        }

        // Move essential buttons into the visible container in whitelist order
        var movedAny = false;
        var fragment = document.createDocumentFragment();
        whitelist.forEach(function(tabId) {
            var btn = essentialBtns[tabId];
            if (!btn) return;
            forceVisible(btn);
            if (btn.parentNode !== visibleContainer) movedAny = true;
            fragment.appendChild(btn);
        });
        if (movedAny) {
            visibleContainer.appendChild(fragment);
        } else {
            // Even if already in place, force visible (Gradio may hide them responsively)
            whitelist.forEach(function(tabId) {
                var btn = essentialBtns[tabId];
                if (btn) forceVisible(btn);
            });
        }

        // Hide the overflow trigger (three-dots) now that essential buttons are visible
        if (overflowMenu) {
            overflowMenu.style.setProperty('display', 'none', 'important');
        }
        return Object.keys(essentialBtns).length > 0;
    }

    // Poll repeatedly to catch late-rendered overflow buttons (Svelte async)
    function ensureOverflowTabsFixed() {
        var attempts = 0;
        var maxAttempts = 30;
        function tick() {
            attempts++;
            var ok = fixOverflowTabs();
            if (ok) return; // moved at least one, done
            if (attempts >= maxAttempts) return;
            setTimeout(tick, 200);
        }
        tick();
        // Also re-run on window resize (Gradio may re-collapse tabs)
        window.addEventListener('resize', function() {
            setTimeout(fixOverflowTabs, 150);
        });
    }

    // ============================================================
    //  Initialize
    // ============================================================

    function tryInit() {
        // Migrate: remove old localStorage key from previous implementation
        try { localStorage.removeItem('sd-hidden-tabs'); } catch (e) { /* ignore */ }

        var tabsEl = document.querySelector('#tabs');
        if (!tabsEl) return false;

        // Collect tabs first — in Gradio 6.x the #tabs element exists early but
        // the tablist/buttons are rendered asynchronously by Svelte. We must
        // wait until at least some tab buttons are present before building the
        // sidebar, otherwise the "插件" section will be empty.
        collectTabs();
        if (allTabPanels.length === 0) {
            // Tab buttons not yet rendered — will retry
            return false;
        }

        var initialized = false;
        if (document.getElementById('sd-sidebar')) {
            // Sidebar already exists — check if we need to rebuild with new tabs
            var sidebar = document.getElementById('sd-sidebar');
            var pluginSection = sidebar.querySelector('[data-section="plugins"]');
            var hasPluginItems = pluginSection && pluginSection.querySelectorAll('.sd-sidebar-item').length > 0;
            var externalCount = allTabPanels.filter(function(p) { return !isBuiltinTab(p) && p.id !== 'sd_webui_agent'; }).length;
            if (!hasPluginItems && externalCount > 0) {
                // Rebuild sidebar HTML to include newly discovered plugin tabs
                rebuildSidebarHTML();
            }
            initialized = true;
        } else {
            injectSidebar();
            setupSidebarEvents();
            observeTabChanges();
            // Hide embedded accordions (ControlNet, ADetailer, etc.) by default
            hideEmbeddedAccordions();
            // FAILSAFE: remove sd-panel-hidden from ALL tab panels.
            document.querySelectorAll('[id^="tab_"]').forEach(function(panel) {
                panel.classList.remove('sd-panel-hidden');
            });
            // Apply user's saved tab visibility preferences from localStorage
            applyAllTabVisibility();
            initialized = true;
        }
        // Always enhance seed dice buttons (function has guard for single execution)
        if (initialized) {
            enhanceSeedDiceButtons();
            relabelControls();
            // Re-apply tab visibility in case Gradio re-rendered tab buttons
            applyAllTabVisibility();
            return true;
        }
        return false;
    }

    // ============================================================
    //  PS-style floating popup for seed row & prompt presets
    //  Elements stay in their original Gradio DOM positions.
    //  CSS toggles display:none / position:fixed floating.
    //  Click sidebar item -> toggle popup. Click outside -> close.
    // ============================================================

    var activePopup = null;
    var documentClickBound = false;

    // Detect which Gradio tab is currently active (txt2img or img2img)
    function getActiveTab() {
        var activeBtn = document.querySelector('#tabs button[aria-selected="true"]');
        if (activeBtn) {
            var controls = activeBtn.getAttribute('aria-controls') || '';
            if (controls.indexOf('img2img') !== -1) return 'img2img';
            if (controls.indexOf('txt2img') !== -1) return 'txt2img';
        }
        var txtPanel = document.getElementById('tab_txt2img');
        var imgPanel = document.getElementById('tab_img2img');
        if (imgPanel && getComputedStyle(imgPanel).display !== 'none') return 'img2img';
        return 'txt2img';
    }

    // Get the popup target element for a given popup type and active tab.
    // Only seed uses PS-style popup now; styles restored to original position.
    function getPopupTarget(popupType, activeTab) {
        if (popupType === 'seed') {
            return document.getElementById(activeTab + '_seed_row');
        }
        return null;
    }

    function setupPopupControls() {
        // Only seed rows use PS-style popup now. Styles row restored to original position.
        ['txt2img', 'img2img'].forEach(function(tab) {
            var seedEl = document.getElementById(tab + '_seed_row');
            if (seedEl) seedEl.classList.add('sd-popup-target');
        });
        // Bind document click for click-away-to-close (once)
        if (!documentClickBound) {
            documentClickBound = true;
            document.addEventListener('click', function(e) {
                if (!activePopup) return;
                if (e.target.closest('[data-popup]')) return;
                // Don't close if click is inside an active popup panel
                var visibleEl = document.querySelector('.sd-popup-target.sd-popup-visible');
                if (visibleEl && visibleEl.contains(e.target)) return;
                closePopup();
            });
        }
    }

    // Relabel common controls to shorter Chinese names to save UI space.
    // Uses elem_id-based targeting to avoid affecting other CFG-related labels.
    // A MutationObserver re-applies labels if Svelte re-renders the components.
    function relabelControls() {
        var labelMap = {
            'txt2img_cfg_scale': 'CFG',
            'img2img_cfg_scale': 'CFG',
            'txt2img_sampling': '采样器',
            'img2img_sampling': '采样器',
            'txt2img_scheduler': '调度器',
            'img2img_scheduler': '调度器'
        };

        function applyLabels() {
            Object.keys(labelMap).forEach(function(id) {
                var el = document.getElementById(id);
                if (!el) return;
                var span = el.querySelector('label span') || el.querySelector('span[data-testid="block-info"]');
                if (span && span.textContent.trim() !== labelMap[id]) {
                    span.textContent = labelMap[id];
                }
            });
        }

        applyLabels();

        // Re-apply on Svelte re-renders
        if (!window.sdRelabelObserver) {
            var observer = new MutationObserver(function(mutations) {
                for (var i = 0; i < mutations.length; i++) {
                    if (mutations[i].type === 'childList' && mutations[i].addedNodes.length > 0) {
                        applyLabels();
                        break;
                    }
                }
            });
            observer.observe(document.body, { childList: true, subtree: true });
            window.sdRelabelObserver = observer;
        }
    }

    // Enhance seed dice button: instead of setting -1 (Gradio default),
    // generate and display an actual random number immediately.
    // Uses document-level event delegation + polling to reliably override
    // Gradio/Svelte's reactive -1 set.
    function enhanceSeedDiceButtons() {
        if (window.sdDiceEnhanced) return;
        window.sdDiceEnhanced = true;

        function setRandomSeed(tab) {
            var row = document.getElementById(tab + '_seed_row');
            if (!row) return;
            var input = row.querySelector('input[type="number"]');
            if (!input) return;
            var randomSeed = Math.floor(Math.random() * 4294967295) + 1;
            var seedStr = String(randomSeed);
            var setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            setter.call(input, seedStr);
            // Use InputEvent with data property — Svelte listens to this for state updates
            input.dispatchEvent(new InputEvent('input', { bubbles: true, cancelable: true, data: seedStr, inputType: 'insertText' }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
        }

        function overrideWhenMinusOne(tab) {
            // Poll multiple times to fight Svelte's reactive overrides
            var attempts = 0;
            var maxAttempts = 20;
            var interval = setInterval(function() {
                attempts++;
                var row = document.getElementById(tab + '_seed_row');
                if (!row) { clearInterval(interval); return; }
                var input = row.querySelector('input[type="number"]');
                if (!input) { clearInterval(interval); return; }
                if (input.value === '-1') {
                    setRandomSeed(tab);
                } else if (attempts >= maxAttempts) {
                    clearInterval(interval);
                }
                if (attempts >= maxAttempts) clearInterval(interval);
            }, 100);
        }

        document.addEventListener('click', function(e) {
            var dice = e.target.closest('#txt2img_random_seed, #img2img_random_seed');
            if (!dice) return;
            var tab = dice.id.indexOf('txt2img') !== -1 ? 'txt2img' : 'img2img';
            // Let Gradio's handler run first (sets -1), then poll to override
            setTimeout(function() { overrideWhenMinusOne(tab); }, 50);
        });
    }

    function togglePopup(popupType, sidebarItem) {
        if (activePopup === popupType) {
            closePopup();
            return;
        }
        closePopup();
        activePopup = popupType;
        document.body.classList.add('sd-popup-' + popupType + '-open');
        sidebarItem.classList.add('popup-active');

        var activeTab = getActiveTab();
        var el = getPopupTarget(popupType, activeTab);
        if (el) {
            el.classList.add('sd-popup-visible');
            if (!el.querySelector('.sd-popup-close')) {
                var closeBtn = document.createElement('button');
                closeBtn.className = 'sd-popup-close';
                closeBtn.innerHTML = '✕';
                closeBtn.title = '关闭';
                closeBtn.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    closePopup();
                });
                el.style.position = 'relative';
                el.appendChild(closeBtn);
            }

        }
    }

    function closePopup() {
        if (!activePopup) return;
        document.body.classList.remove('sd-popup-' + activePopup + '-open');
        // Hide all visible popup elements
        document.querySelectorAll('.sd-popup-target.sd-popup-visible').forEach(function(el) {
            el.classList.remove('sd-popup-visible');
        });
        var sidebar = document.getElementById('sd-sidebar');
        if (sidebar) {
            sidebar.querySelectorAll('.sd-sidebar-item.popup-active').forEach(function(el) {
                el.classList.remove('popup-active');
            });
        }
        activePopup = null;
    }

    function init() {
        if (tryInit()) {
            // Light periodic: re-mark popup targets after Gradio re-renders
            var markCount = 0;
            var markInterval = setInterval(function() {
                markCount++;
                setupPopupControls();
                if (markCount >= 15) clearInterval(markInterval);
            }, 1000);
            return;
        }
        var retries = 0;
        // Gradio 6.x UI creation can take 40+ seconds on first load (Svelte
        // async rendering). Retry for up to 90 seconds to make sure all tab
        // buttons are present before giving up.
        var maxRetries = 180;
        var checkInterval = setInterval(function() {
            retries++;
            if (tryInit() || retries >= maxRetries) {
                clearInterval(checkInterval);
            }
        }, 500);
    }

    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        init();
    } else {
        document.addEventListener('DOMContentLoaded', init);
    }

})();

// ============================================================
//  Theme Manager - Apple-style multi-theme system
//  参考电商助手的主题切换实现
// ============================================================
(function() {
    'use strict';

    var THEMES = [
        { id: 'theme-apple-light', name: '苹果浅色', swatch: 'linear-gradient(135deg, #0071e3 50%, #f5f5f7 50%)' },
        { id: 'theme-dark-blue', name: '暗夜蓝', swatch: 'linear-gradient(135deg, #0a84ff 50%, #1c1c28 50%)' },
        { id: 'theme-dark-green', name: '暗夜绿', swatch: 'linear-gradient(135deg, #30d158 50%, #1c1c20 50%)' },
        { id: 'theme-dark-purple', name: '暗夜紫', swatch: 'linear-gradient(135deg, #bf5af2 50%, #221f2e 50%)' },
        { id: 'theme-dark-orange', name: '暗夜橙', swatch: 'linear-gradient(135deg, #ff9500 50%, #1c1c1e 50%)' },
        { id: 'theme-obsidian', name: '黑曜石', swatch: 'linear-gradient(135deg, #f5f5f7 50%, #111111 50%)' },
        { id: 'theme-light-blue', name: '纯净蓝', swatch: 'linear-gradient(135deg, #0a84ff 50%, #e8f0fe 50%)' },
        { id: 'theme-light-green', name: '纯净绿', swatch: 'linear-gradient(135deg, #34c759 50%, #eaf5eb 50%)' },
        { id: 'theme-warm-orange', name: '暖阳橙', swatch: 'linear-gradient(135deg, #ff9500 50%, #f5ede0 50%)' },
        { id: 'theme-sakura-pink', name: '樱花粉', swatch: 'linear-gradient(135deg, #e84393 50%, #f5e8ee 50%)' },
    ];

    var STORAGE_KEY = 'sd-webui-theme';
    var DEFAULT_THEME = 'theme-apple-light';

    function getCurrentTheme() {
        try {
            var saved = localStorage.getItem(STORAGE_KEY);
            if (saved && THEMES.some(function(t) { return t.id === saved; })) {
                return saved;
            }
        } catch (e) {}
        return DEFAULT_THEME;
    }

    function applyTheme(themeId) {
        THEMES.forEach(function(t) {
            document.body.classList.remove(t.id);
            document.documentElement.classList.remove(t.id);
        });
        document.body.classList.add(themeId);
        document.documentElement.classList.add(themeId);
        try { localStorage.setItem(STORAGE_KEY, themeId); } catch (e) {}
        updateThemeModalActive(themeId);
    }

    // Gradio's Svelte app may overwrite body classList on mount/re-render,
    // which strips our theme class and causes the theme to revert on restart.
    // This guard re-applies the saved theme whenever the class is removed.
    function ensureThemePersisted() {
        var current = getCurrentTheme();
        if (!document.body.classList.contains(current)) {
            document.body.classList.add(current);
        }
        if (!document.documentElement.classList.contains(current)) {
            document.documentElement.classList.add(current);
        }
    }

    function setupThemePersistenceGuard() {
        if (window.__sdThemeGuardSet) return;
        window.__sdThemeGuardSet = true;
        ensureThemePersisted();

        // Watch body class changes and restore theme if stripped by Svelte
        var observer = new MutationObserver(function(mutations) {
            for (var i = 0; i < mutations.length; i++) {
                if (mutations[i].attributeName === 'class') {
                    ensureThemePersisted();
                    break;
                }
            }
        });
        observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });

        // Poll briefly during startup to fight Svelte's initial mount overwrite
        var polls = 0;
        var interval = setInterval(function() {
            ensureThemePersisted();
            polls++;
            if (polls >= 20) clearInterval(interval);
        }, 500);
    }

    function createThemeModal() {
        if (document.getElementById('sd-theme-overlay')) return;

        var overlay = document.createElement('div');
        overlay.id = 'sd-theme-overlay';

        var modal = document.createElement('div');
        modal.id = 'sd-theme-modal';

        var html = '';
        html += '<button id="sd-theme-close" title="关闭">✕</button>';
        html += '<h3>🎨 主题切换</h3>';
        html += '<p class="sd-theme-subtitle">选择你喜欢的 UI 主题风格</p>';
        html += '<div class="sd-theme-grid">';

        THEMES.forEach(function(theme) {
            html += '<div class="sd-theme-option" data-theme="' + theme.id + '">';
            html += '  <div class="sd-theme-swatch" style="background: ' + theme.swatch + ';"></div>';
            html += '  <span>' + theme.name + '</span>';
            html += '</div>';
        });

        html += '</div>';
        modal.innerHTML = html;
        overlay.appendChild(modal);
        document.body.appendChild(overlay);

        var closeBtn = document.getElementById('sd-theme-close');
        if (closeBtn) closeBtn.addEventListener('click', hideThemeModal);

        overlay.addEventListener('click', function(e) {
            if (e.target === overlay) hideThemeModal();
        });

        modal.querySelectorAll('.sd-theme-option').forEach(function(opt) {
            opt.addEventListener('click', function() {
                var themeId = this.dataset.theme;
                applyTheme(themeId);
                setTimeout(hideThemeModal, 200);
            });
        });

        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') hideThemeModal();
        });
    }

    function updateThemeModalActive(themeId) {
        var options = document.querySelectorAll('.sd-theme-option');
        options.forEach(function(opt) {
            opt.classList.toggle('sd-theme-active', opt.dataset.theme === themeId);
        });
    }

    function showThemeModal() {
        createThemeModal();
        var overlay = document.getElementById('sd-theme-overlay');
        if (overlay) {
            overlay.classList.add('sd-theme-visible');
            updateThemeModalActive(getCurrentTheme());
        }
    }

    function hideThemeModal() {
        var overlay = document.getElementById('sd-theme-overlay');
        if (overlay) overlay.classList.remove('sd-theme-visible');
    }

    function addThemeButtonToSidebar() {
        var footer = document.querySelector('.sd-sidebar-footer');
        if (!footer) return;

        // Add "主页" (Home) button above theme button — switches to txt2img tab
        if (!document.getElementById('sd-home-btn')) {
            var homeBtn = document.createElement('div');
            homeBtn.className = 'sd-sidebar-footer-item';
            homeBtn.id = 'sd-home-btn';
            homeBtn.innerHTML = '<span class="sd-sidebar-footer-icon">🏠</span><span>主页</span>';
            homeBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                // Inline switch logic (self-contained — switchTab is in a
                // different IIFE scope and inaccessible here)
                var txt2imgBtn = document.querySelector('button[aria-controls="tab_txt2img"]');
                if (txt2imgBtn) txt2imgBtn.click();
                // Restore native panels, hide non-native ones
                var nativeIds = ['txt2img', 'img2img', 'extras', 'pnginfo', 'tutorial_center', 'settings', 'extensions'];
                document.querySelectorAll('[id^="tab_"]').forEach(function(panel) {
                    var id = panel.id.replace('tab_', '');
                    if (nativeIds.indexOf(id) === -1) {
                        panel.classList.add('sd-panel-hidden');
                        panel.classList.remove('sd-panel-active');
                    } else {
                        panel.classList.remove('sd-panel-hidden');
                        panel.classList.remove('sd-panel-active');
                    }
                });
                // Update sidebar highlight
                document.querySelectorAll('.sd-sidebar-item, .sd-sidebar-footer-item').forEach(function(el) {
                    el.classList.remove('active');
                });
                homeBtn.classList.add('active');
            });
            footer.appendChild(homeBtn);
        }

        // Add theme button
        if (!document.getElementById('sd-theme-btn')) {
            var themeBtn = document.createElement('div');
            themeBtn.className = 'sd-sidebar-footer-item';
            themeBtn.id = 'sd-theme-btn';
            themeBtn.innerHTML = '<span class="sd-sidebar-footer-icon">🎨</span><span>主题</span>';
            themeBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                showThemeModal();
            });
            footer.appendChild(themeBtn);
        }
    }

    function initThemeManager() {
        applyTheme(getCurrentTheme());
        setupThemePersistenceGuard();

        var retries = 0;
        var maxRetries = 40;
        var checkInterval = setInterval(function() {
            retries++;
            if (document.querySelector('.sd-sidebar-footer') || retries >= maxRetries) {
                clearInterval(checkInterval);
                addThemeButtonToSidebar();
            }
        }, 500);
    }

    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        initThemeManager();
    } else {
        document.addEventListener('DOMContentLoaded', initThemeManager);
    }

    window.SDThemeManager = {
        apply: applyTheme,
        current: getCurrentTheme,
        show: showThemeModal,
        hide: hideThemeModal,
        themes: THEMES,
        rebuildFooter: addThemeButtonToSidebar
    };

})();

// Close outer IIFE
})();