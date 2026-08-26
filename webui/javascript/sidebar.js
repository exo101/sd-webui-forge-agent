/**
 * Sidebar Navigation for SD WebUI
 * Apple-style sidebar - the ONLY way to navigate tabs.
 * All Gradio tab panels are hidden by default.
 * Clicking a sidebar item shows the corresponding panel.
 */

(function() {
    'use strict';

    // ============================================================
    //  Sidebar Configuration (User-defined modules)
    // ============================================================

    // Map: custom label -> Gradio tab ID
    // 经过实际检查各扩展注册的 tab ID
    // 注意: tabId 为 'txt2img' 的项是 webui 内置页面, 始终存在
    //       其他 tabId 对应扩展插件, 插件未安装时标签页不存在, 侧边栏应自动隐藏
    var sidebarModules = [
        {
            name: '海报设计',
            icon: '🎨',
            items: [
                { label: '海报设计工作台', tabId: 'poster_design_tab' },
            ]
        },
        {
            name: '图层处理',
            icon: '◈',
            items: [
                { label: '点选分割', tabId: 'Segmentation_Tab', subTabLabel: '点选分割' },
                { label: '智能抠图', tabId: 'Segmentation_Tab', subTabLabel: '智能抠图' },
                { label: '图层分离', tabId: 'Segmentation_Tab', subTabLabel: '图层分离' },
                { label: '图像清理', tabId: 'Segmentation_Tab', subTabLabel: '图像清理' },
            ]
        },
        {
            name: '视觉分析',
            icon: '◉',
            items: [
                { label: '图像识别', tabId: 'aesthetic_enhancement_tab', subTabLabel: '🖼 图像识别' },
                { label: '打光辅助', tabId: 'aesthetic_enhancement_tab', subTabLabel: '💡 打光辅助' },
                { label: '人物与场景分析', tabId: 'aesthetic_enhancement_tab', subTabLabel: '人物与场景分析' },
                { label: '构图技巧', tabId: 'aesthetic_enhancement_tab', subTabLabel: '📐 构图技巧' },
                { label: '画师百科', tabId: 'aesthetic_enhancement_tab', subTabLabel: '🎨 画师百科' },
                { label: '标签器', tabId: 'tagger' },
            ]
        },
        {
            name: '多媒体视频生成',
            icon: '▶',
            items: [
                { label: '音乐生成', tabId: 'multimodal_media_tab' },
                { label: '视频关键帧', tabId: 'multimodal_media_tab', subTabLabel: '2. 视频关键帧提取' },
                { label: 'MiniMax H3工作台', tabId: 'forge_h3_studio' },
                { label: 'Kling可灵视频生成', tabId: 'multimodal_media_tab', subTabLabel: '3. Kling 可灵视频生成' },
                { label: 'ACE-Step音乐生成', tabId: 'multimodal_media_tab', subTabLabel: '5. ACE-Step 音乐生成' },
                { label: 'Qwen3-TTS 语音合成', tabId: 'multimodal_media_tab', subTabLabel: '1. Qwen3-TTS 语音合成' },
            ]
        },
        {
            name: '参数设置',
            icon: '⚙',
            items: [
                { label: '采样器', tabId: 'txt2img', containerIds: ['sampler_selection_txt2img', 'txt2img_scheduler', 'txt2img_cfg_scale', 'txt2img_distilled_cfg_scale'] },
                { label: '随机种子', tabId: 'txt2img', containerIds: ['txt2img_seed_row'] },
                { label: '关键词预设', tabId: 'txt2img', containerIds: ['txt2img_styles_row', 'txt2img_tools'] },
            ]
        },
        {
            name: '控制辅助',
            icon: '🎯',
            items: [
                { label: '区域控制', tabId: 'txt2img', accordionId: 'RP_maint2i', subTabLabel: '🎯 区域提示' },
                { label: 'ControlNet', tabId: 'txt2img', accordionId: 'controlnet' },
                { label: '多图参考', tabId: 'txt2img', accordionId: 'label:多图参考' },
                { label: '纵横比助手', tabId: 'txt2img', accordionId: 'aspect-ratio-helper-accordion' },
                { label: '通配符', tabId: 'sddp-wildcard-manager' },
                { label: '脚本', tabId: 'txt2img' },
            ]
        },
        {
            name: '模型管理',
            icon: '📦',
            items: [
                { label: '模型下载器', tabId: 'model-downloader' },
                { label: 'Civitai 浏览器', tabId: 'civitai_interface_neo' },
                { label: '模型合并', tabId: 'supermerger' },
            ]
        },
        {
            name: '后期处理',
            icon: '🔧',
            items: [
                { label: '无边图像浏览', tabId: 'infinite-image-browsing' },
                { label: 'SeedVR2 高清放大', tabId: 'extras' },
                { label: 'ADetailer 面部修复', tabId: 'txt2img', accordionId: 'script_txt2img_adetailer_ad_main_accordion' },
                { label: '图像对比', tabId: 'sd-webui-image-comparison' },
                { label: '高清修复 (Hires. fix)', tabId: 'txt2img', accordionId: 'txt2img_hr' },
                { label: '通配符', tabId: 'sddp-wildcard-manager' },
                { label: '图生3D', tabId: 'trellis2_3d_generator' },
            ]
        }
    ];

    // ============================================================
    //  动态检测: 检查标签页是否存在 (插件是否已安装)
    // ============================================================

    // 内置标签页始终存在, 不需要检测
    var builtinTabs = {
        'txt2img': true,
        'img2img': true,
        'extras': true,
        'settings': true,
        'extensions': true,
        'checkpoint': true,
    };

    function tabExists(tabId) {
        // 内置标签页直接返回 true
        if (builtinTabs[tabId]) return true;
        // 检查标签页面板是否存在
        var panel = document.getElementById('tab_' + tabId);
        if (panel) return true;
        // 检查标签按钮是否存在
        var btn = document.querySelector('button[aria-controls="tab_' + tabId + '"]');
        return !!btn;
    }

    // ============================================================
    //  Collect all Gradio tabs dynamically
    // ============================================================

    var allTabPanels = []; // {id, label, elem}

    function collectTabs() {
        allTabPanels = [];
        var panels = document.querySelectorAll('[id^="tab_"]');
        panels.forEach(function(panel) {
            var id = panel.id.replace('tab_', '');
            var label = id; // fallback
            // Try to find the tab label text
            var btn = document.querySelector('button[aria-controls="tab_' + id + '"]');
            if (btn) {
                label = btn.textContent.trim();
            }
            allTabPanels.push({ id: id, label: label, elem: panel });
        });
    }

    // ============================================================
    //  Panel visibility control
    // ============================================================

    function showOnlyPanel(panelId) {
        allTabPanels.forEach(function(panel) {
            var isActive = panel.id === panelId;
            // Use CSS class to toggle visibility
            panel.elem.classList.toggle('sd-panel-active', isActive);
            panel.elem.classList.toggle('sd-panel-hidden', !isActive);
            // Also set display style directly for reliability
            panel.elem.style.display = isActive ? '' : 'none';
        });
    }

    // ============================================================
    //  Embedded plugin accordion visibility control
    //  These accordions are embedded in txt2img/img2img pages
    //  and should be hidden by default, only shown when clicked
    //  from the sidebar.
    // ============================================================

    // IDs of parameter containers that can be shown/hidden individually
    // These are standard Gradio components (sampler, seed, scheduler, styles)
    // that are always visible in txt2img but can be isolated when clicked from sidebar
    var parameterContainerIds = [
        'sampler_selection_txt2img',
        'txt2img_scheduler',
        'txt2img_seed_row',
        'txt2img_styles_row',
        'txt2img_tools',
    ];

    function hideAllParameterContainers() {
        parameterContainerIds.forEach(function(id) {
            var el = document.getElementById(id);
            if (el) {
                el.style.display = 'none';
            }
        });
    }

    function showParameterContainers(ids) {
        // First hide all parameter containers
        hideAllParameterContainers();
        // Then show only the specified ones
        ids.forEach(function(id) {
            var el = document.getElementById(id);
            if (el) {
                el.style.display = '';
            }
        });
    }

    // IDs of embedded accordions that should be hidden by default
    // These are rendered as AlwaysVisible scripts inside txt2img/img2img pages
    var embeddedAccordionIds = [
        'aspect-ratio-helper-accordion',   // 纵横比助手
        'txt2img_hr',                      // Hires. fix 高分辨率修复
        'sddp-dynamic-prompting',          // Dynamic Prompts
        // ADetailer - different IDs for txt2img and img2img
        'script_txt2img_adetailer_ad_main_accordion',
        'script_img2img_adetailer_ad_main_accordion',
        // Regional Prompter - different IDs for txt2img and img2img
        'RP_maint2i',
        'RP_maini2i',
        // ControlNet integrated
        'controlnet',
    ];

    // Also hide the entire script containers (built-in scripts like ControlNet, 多图参考, Torch编译集成, 脚本)
    // These are part of the core UI and contain AlwaysVisible scripts
    var scriptContainerIds = [
        'txt2img_script_container',
        'img2img_script_container',
    ];

    function hideEmbeddedAccordions() {
        // Hide individual accordions by element ID
        embeddedAccordionIds.forEach(function(id) {
            var el = document.getElementById(id);
            if (el) {
                var container = el.closest('.group') || el.parentElement;
                if (container) {
                    container.style.display = 'none';
                } else {
                    el.style.display = 'none';
                }
            }
        });
        // Hide input-accordion based scripts (多图参考, soft-inpainting, radial, pid, etc.)
        // These use InputAccordion which generates auto-increment element IDs
        document.querySelectorAll('.input-accordion').forEach(function(el) {
            var container = el.closest('.group') || el.parentElement;
            if (container) {
                container.style.display = 'none';
            } else {
                el.style.display = 'none';
            }
        });
        // Hide script containers (core built-in features)
        scriptContainerIds.forEach(function(id) {
            var el = document.getElementById(id);
            if (el) {
                el.style.display = 'none';
            }
        });
    }

    function showEmbeddedAccordion(id) {
        // First hide all embedded accordions
        hideEmbeddedAccordions();
        // Show the script container (txt2img/img2img)
        // The accordion is inside one of the script containers
        var el = document.getElementById(id);
        // If not found by ID, try searching by label text (e.g. "label:多图参考")
        if (!el && id && id.startsWith('label:')) {
            var labelText = id.substring(6);
            el = document.querySelector('.input-accordion .label-wrap span')?.closest('.input-accordion');
            if (el) {
                var allAccordions = document.querySelectorAll('.input-accordion');
                for (var i = 0; i < allAccordions.length; i++) {
                    var labelEl = allAccordions[i].querySelector('.label-wrap span');
                    if (labelEl && labelEl.textContent.trim() === labelText) {
                        el = allAccordions[i];
                        break;
                    }
                }
            }
        }
        if (el) {
            // Find the nearest script container and show it
            var scriptContainer = el.closest('#txt2img_script_container, #img2img_script_container');
            if (scriptContainer) {
                scriptContainer.style.display = '';
            }
            // Show the accordion container
            var container = el.closest('.group') || el.parentElement;
            if (container) {
                container.style.display = '';
            } else {
                el.style.display = '';
            }
            // Click the label-wrap to expand the accordion (important for InputAccordion)
            var labelWrap = el.querySelector('.label-wrap');
            if (labelWrap) {
                labelWrap.dispatchEvent(new MouseEvent('click', {
                    bubbles: true,
                    cancelable: true,
                    view: window
                }));
            }
        }
    }

    // ============================================================
    //  Switch to a tab
    // ============================================================

    function switchTab(tabId, showAccordionId, subTabLabel, containerIds) {
        if (!tabId) return;

        // Click the Gradio tab button to ensure Gradio internal state is updated
        var tabButton = document.querySelector('button[aria-controls="tab_' + tabId + '"]');
        if (tabButton) {
            tabButton.click();
        }

        // Show the selected panel, hide all others
        showOnlyPanel(tabId);

        // Show/hide parameter containers (sampler, scheduler, seed, styles)
        // These are hidden by default; only shown when a specific containerIds is provided
        if (containerIds && containerIds.length > 0) {
            showParameterContainers(containerIds);
        } else {
            hideAllParameterContainers();
        }

        // Show/hide embedded accordions
        if (showAccordionId) {
            showEmbeddedAccordion(showAccordionId);
        } else if (tabId === 'txt2img' || tabId === 'img2img') {
            // First hide all embedded accordions (ControlNet, 多图参考, 纵横比助手, etc.)
            hideEmbeddedAccordions();
            // Then show the script containers (so 脚本 section is visible)
            var mainContainers = ['txt2img_script_container', 'img2img_script_container'];
            mainContainers.forEach(function(id) {
                var el = document.getElementById(id);
                if (el) el.style.display = '';
            });
        } else {
            hideEmbeddedAccordions();
        }

        // Click sub-tab inside the panel if specified
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
//  Hide/Show item state (localStorage)
// ============================================================

function getHideKey(modName, itemLabel) {
    return 'sd_hide_' + modName + '_' + itemLabel;
}

function loadHiddenItems() {
    var hidden = {};
    try {
        for (var key in localStorage) {
            if (key.startsWith('sd_hide_') && localStorage.getItem(key) === 'true') {
                hidden[key] = true;
            }
        }
    } catch (e) {}
    return hidden;
}

function saveHideState(key, hidden) {
    try {
        localStorage.setItem(key, hidden ? 'true' : 'false');
    } catch (e) {}
}

// ============================================================
//  Inject Sidebar HTML
// ============================================================

function injectSidebar() {
    if (document.getElementById('sd-sidebar')) return;

    // Hide Gradio tab buttons via JavaScript for reliability
    var tabsContainer = document.querySelector('#tabs');
    if (tabsContainer) {
        // Find and hide tab navigation buttons
        var tabNav = tabsContainer.querySelector(':scope > .tab-nav, :scope > div[role="tablist"]');
        if (tabNav) {
            tabNav.style.display = 'none';
        } else {
            // Fallback: hide all buttons that are direct children of #tabs
            // (excluding buttons inside tab panels)
            var children = tabsContainer.children;
            for (var i = 0; i < children.length; i++) {
                var child = children[i];
                if (child.tagName === 'BUTTON' || (child.tagName === 'DIV' && child.querySelector('button'))) {
                    // Check if this is a tab nav (has buttons, no id starting with "tab_")
                    if (child.tagName === 'BUTTON' || !child.id || !child.id.startsWith('tab_')) {
                        if (child.tagName === 'BUTTON' || child.querySelector('button')) {
                            child.style.display = 'none';
                        }
                    }
                }
            }
        }
    }

    // Collect all tabs first
    collectTabs();

    // Load hidden items state
    var hiddenItems = loadHiddenItems();

    // Build sidebar
    var sidebar = document.createElement('aside');
    sidebar.id = 'sd-sidebar';
    sidebar.className = 'sd-sidebar';

    var html = '';
    html += '<div class="sd-sidebar-brand">';
    html += '  <div class="sd-sidebar-logo">SD</div>';
    html += '  <div class="sd-sidebar-title">';
    html += '    <h1>Stable Diffusion</h1>';
    html += '    <p>Forge NEO</p>';
    html += '  </div>';
    html += '</div>';
    html += '<nav class="sd-sidebar-nav">';

    // Custom modules - 动态过滤: 只渲染已安装插件的标签页
    sidebarModules.forEach(function(mod, mi) {
        // 过滤掉 tabId 不存在 (插件未安装) 的项
        var visibleItems = mod.items.filter(function(item) {
            return tabExists(item.tabId);
        });
        // 如果整组都没有安装, 跳过整个分区
        if (visibleItems.length === 0) return;

        var expanded = '';
        html += '<div class="sd-sidebar-section' + expanded + '">';
        html += '  <div class="sd-sidebar-section-header" data-section="' + mi + '">';
        html += '    <span class="sd-sidebar-section-icon">' + mod.icon + '</span>';
        html += '    <span class="sd-sidebar-section-label">' + mod.name + '</span>';
        html += '    <span class="sd-sidebar-chevron">▾</span>';
        html += '  </div>';
        html += '  <div class="sd-sidebar-section-body">';
        visibleItems.forEach(function(item) {
            var hideKey = getHideKey(mod.name, item.label);
            var isHidden = hiddenItems[hideKey] === true;
            html += '    <div class="sd-sidebar-item' + (isHidden ? ' sd-item-hidden' : '') + '" data-tab="' + item.tabId + '" data-accordion="' + (item.accordionId || '') + '" data-subtab="' + (item.subTabLabel || '') + '" data-containers="' + (item.containerIds ? item.containerIds.join(',') : '') + '" data-hide-key="' + hideKey + '">';
            html += '      <span class="sd-sidebar-item-label">' + item.label + '</span>';
            html += '      <span class="sd-sidebar-item-eye" title="点击隐藏/显示">👁</span>';
            html += '    </div>';
        });
        html += '  </div>';
        html += '</div>';
    });

    html += '</nav>';

    // Footer
        html += '<div class="sd-sidebar-footer">';
        html += '  <div class="sd-sidebar-footer-item" data-tab="txt2img">';
        html += '    <span class="sd-sidebar-footer-icon">🏠</span>';
        html += '    <span>主页</span>';
        html += '  </div>';
        html += '  <div class="sd-sidebar-footer-item" id="sd-restore-btn" title="恢复隐藏的模块">';
        html += '    <span class="sd-sidebar-footer-icon">👁</span>';
        html += '    <span>恢复隐藏项</span>';
        html += '  </div>';
        html += '  <div class="sd-sidebar-footer-item" data-tab="settings">';
        html += '    <span class="sd-sidebar-footer-icon">⚙</span>';
        html += '    <span>设置</span>';
        html += '  </div>';
        html += '  <div class="sd-sidebar-footer-item" data-tab="extensions">';
        html += '    <span class="sd-sidebar-footer-icon">🧩</span>';
        html += '    <span>扩展</span>';
        html += '  </div>';
        html += '</div>';

    sidebar.innerHTML = html;

        // Toggle button
        var toggle = document.createElement('button');
        toggle.id = 'sd-sidebar-toggle';
        toggle.className = 'sd-sidebar-toggle';
        toggle.innerHTML = '☰';
        toggle.setAttribute('title', '切换侧边栏');

        document.body.appendChild(sidebar);
        document.body.appendChild(toggle);
        document.body.classList.add('sd-sidebar-active');
    }

    // ============================================================
    //  Setup events
    // ============================================================

    function setupSidebarEvents() {
        var sidebar = document.getElementById('sd-sidebar');
        if (!sidebar) return;

        // Section header expand/collapse
        sidebar.querySelectorAll('.sd-sidebar-section-header').forEach(function(header) {
            header.addEventListener('click', function(e) {
                e.stopPropagation();
                this.closest('.sd-sidebar-section').classList.toggle('expanded');
            });
        });

        // Sidebar item click -> switch tab
        sidebar.querySelectorAll('.sd-sidebar-item, .sd-sidebar-footer-item').forEach(function(item) {
            item.addEventListener('click', function(e) {
                // Ignore click on the eye icon
                if (e.target.classList.contains('sd-sidebar-item-eye')) return;
                e.stopPropagation();

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

        // Eye icon click -> toggle hide/show item
        sidebar.querySelectorAll('.sd-sidebar-item-eye').forEach(function(eye) {
            eye.addEventListener('click', function(e) {
                e.stopPropagation();
                var item = this.closest('.sd-sidebar-item');
                if (!item) return;
                var hideKey = item.dataset.hideKey;
                var isHidden = item.classList.toggle('sd-item-hidden');
                saveHideState(hideKey, isHidden);
                // If we're in restore mode and an item is restored, exit restore mode
                if (!isHidden && sidebar.classList.contains('sd-restore-mode')) {
                    // Check if any hidden items remain
                    var remaining = sidebar.querySelectorAll('.sd-sidebar-item.sd-item-hidden');
                    if (remaining.length === 0) {
                        sidebar.classList.remove('sd-restore-mode');
                    }
                }
            });
        });

        // Restore button - show/hide restore mode
        var restoreBtn = document.getElementById('sd-restore-btn');
        if (restoreBtn) {
            restoreBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                var sidebar = document.getElementById('sd-sidebar');
                if (!sidebar) return;
                sidebar.classList.toggle('sd-restore-mode');
                // Update button text
                if (sidebar.classList.contains('sd-restore-mode')) {
                    this.querySelector('span:last-child').textContent = '退出管理';
                } else {
                    this.querySelector('span:last-child').textContent = '恢复隐藏项';
                }
            });
        }

        // Toggle sidebar
        var toggle = document.getElementById('sd-sidebar-toggle');
        if (toggle) {
            toggle.addEventListener('click', function() {
                document.body.classList.toggle('sd-sidebar-collapsed');
            });
        }
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

        // Observe new tabs
        var bodyObserver = new MutationObserver(function() {
            var newTabs = document.querySelectorAll('[id^="tab_"]:not([data-sidebar-observed])');
            newTabs.forEach(function(container) {
                container.setAttribute('data-sidebar-observed', 'true');
                var btn = container.querySelector('button');
                if (btn) {
                    observer.observe(btn, { attributes: true, attributeFilter: ['aria-selected'] });
                }
            });
        });
        bodyObserver.observe(document.body, { childList: true, subtree: true });
    }

    // ============================================================
    //  Initialize
    // ============================================================

    // 重新构建侧边栏: 当 Gradio 稍晚渲染扩展标签页时, 补上之前隐藏的入口
    function rebuildSidebarIfNeeded() {
        var sidebar = document.getElementById('sd-sidebar');
        if (!sidebar) return;

        // 检查是否有之前不存在但现在已加载的标签页
        var needsRebuild = false;
        sidebarModules.forEach(function(mod) {
            mod.items.forEach(function(item) {
                if (tabExists(item.tabId)) {
                    // 该标签页已存在, 检查侧边栏中是否已有对应项
                    var subtab = item.subTabLabel || '';
                    var existing = sidebar.querySelector(
                        '.sd-sidebar-item[data-tab="' + item.tabId + '"]' +
                        '[data-subtab="' + subtab + '"]'
                    );
                    if (!existing) {
                        needsRebuild = true;
                    }
                }
            });
        });

        if (!needsRebuild) return;

        // 移除旧侧边栏并重新注入
        sidebar.remove();
        var oldToggle = document.getElementById('sd-sidebar-toggle');
        if (oldToggle) oldToggle.remove();
        document.body.classList.remove('sd-sidebar-active');

        if (document.querySelector('#tabs')) {
            injectSidebar();
            setupSidebarEvents();
            observeTabChanges();
            collectTabs();
            hideEmbeddedAccordions();
            hideAllParameterContainers();
            var newSidebar = document.getElementById('sd-sidebar');
            if (newSidebar) {
                var homeBtn = newSidebar.querySelector('.sd-sidebar-footer-item[data-tab="txt2img"]');
                if (homeBtn) homeBtn.classList.add('active');
            }
        }
    }

    function tryInit() {
        if (document.getElementById('sd-sidebar')) return true;
        if (document.querySelector('#tabs')) {
            injectSidebar();
            setupSidebarEvents();
            observeTabChanges();
            // Show home page (txt2img) by default, but hide all embedded accordions
            collectTabs();
            showOnlyPanel('txt2img');
            hideEmbeddedAccordions();
            hideAllParameterContainers();
            // Activate home button
            var sidebar = document.getElementById('sd-sidebar');
            if (sidebar) {
                var homeBtn = sidebar.querySelector('.sd-sidebar-footer-item[data-tab="txt2img"]');
                if (homeBtn) homeBtn.classList.add('active');
            }
            // 延迟重新检查: Gradio 可能稍晚渲染扩展标签页 (如异步加载的扩展)
            setTimeout(rebuildSidebarIfNeeded, 2000);
            setTimeout(rebuildSidebarIfNeeded, 5000);
            return true;
        }
        return false;
    }

    function init() {
        if (tryInit()) return;
        var retries = 0;
        var maxRetries = 30;
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