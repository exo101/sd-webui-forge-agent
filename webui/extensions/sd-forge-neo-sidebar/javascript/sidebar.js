// ============================================================
// SD WebUI Forge NEO - Sidebar & Top Toolbar
// Photoshop-like toolbar for plugin management
// ============================================================

(function () {
    'use strict';

    // --- Configuration ---
    const TOPBAR_ITEMS = [
        { id: 'txt2img_switch', icon: '🎨', label: '文生图',              type: 'tab', tabId: 'txt2img' },
        { id: 'img2img_switch', icon: '🖌️', label: '图生图',              type: 'tab', tabId: 'img2img' },
        { id: 'pnginfo',        icon: '🖼️', label: 'PNG信息',              type: 'tab', tabId: 'pnginfo' },
        { id: 'aesthetic',      icon: '🎨', label: '美学提升',              type: 'tab', tabId: 'aesthetic_enhancement_tab' },
        { id: 'vision_chat',    icon: '💬', label: '视觉交互',              type: 'tab', tabId: 'Vision_Chat_Tab' },
        { id: 'lighting',       icon: '💡', label: '打光辅助',              type: 'tab', tabId: 'lighting_assistant' },
        { id: 'settings',       icon: '⚙️', label: '设置',                  type: 'tab', tabId: 'settings' },
        { id: 'extensions',     icon: '🧩', label: '扩展',                  type: 'tab', tabId: 'extensions' },
        { id: 'image_browsing', icon: '🌄', label: '图像浏览',              type: 'tab', tabId: 'infinite-image-browsing' },
        { id: 'camera_angle',   icon: '📷', label: '相机角度',              type: 'tab', tabId: 'camera_angle_selector' },
        { id: 'comparison',     icon: '📊', label: '图像对比',              type: 'tab', tabId: 'sd-webui-image-comparison' },
        { id: 'model_downloader', icon: '📥', label: '模型下载',            type: 'tab', tabId: 'model-downloader' },
        { id: 'tts_voice',     icon: '🎤',  label: '语音合成',              type: 'subtab', tabId: 'multimodal_media_tab', subtabLabel: '1. Qwen3-TTS 语音合成' },
        { id: 'video_keyframe', icon: '🎞️', label: '视频关键帧',            type: 'subtab', tabId: 'multimodal_media_tab', subtabLabel: '3. 视频关键帧提取' },
        { id: 'music_gen',      icon: '🎵',  label: '音乐生成',              type: 'subtab', tabId: 'multimodal_media_tab', subtabLabel: '6. ACE-Step 音乐生成' },
        { id: 'matting',        icon: '✂️',  label: '智能抠图',              type: 'subtab', tabId: 'Segmentation_Tab', subtabLabel: '智能抠图' },
        { id: 'point_seg',      icon: '📍',  label: '点选分割',              type: 'subtab', tabId: 'Segmentation_Tab', subtabLabel: '点选分割' },
        { id: 'cleanup',        icon: '🧹',  label: '图像清理',              type: 'subtab', tabId: 'Segmentation_Tab', subtabLabel: '图像清理' },
        { id: 'trellis',        icon: '🧊', label: '图生3D',                type: 'tab', tabId: 'trellis2_3d_generator' },
        { id: 'tagger',         icon: '🏷️', label: '标签器',                type: 'tab', tabId: 'tagger' },
        { id: 'civitai',        icon: '🏪', label: 'CivitAI',              type: 'tab', tabId: 'civitai_interface_neo' },
        { id: 'supermerger',    icon: '🔀', label: '模型融合',              type: 'tab', tabId: 'supermerger' },
        { id: 'lora',           icon: '💪', label: 'Lora',                type: 'extra_tab', tabIds: ['txt2img_lora', 'img2img_lora'] },
        { id: 'extras',        icon: '📐',  label: '后期处理',              type: 'tab', tabId: 'extras' },
        { id: 'tutorial_center', icon: '📚', label: '教程中心',              type: 'tab', tabId: 'tutorial_center' },
        { id: 'dynamic_prompts', icon: '🃏', label: '通配符',              type: 'accordion', match: 'Dynamic Prompts' },
    ];

    const SIDEBAR_ITEMS = [
        // Group 1: 显存/编译
        { id: 'hires_fix',     icon: '🔍',  label: '高分辨率修复',        type: 'accordion', match: 'Hires. fix' },
        { id: 'torch_compile', icon: '⚡',  label: '编译集成',            type: 'accordion', match: 'Torch 编译集成' },
        { id: 'model_components', icon: '🧠',  label: '生图模型组件',        type: 'model_components' },
        { id: 'sep1',          type: 'separator' },
        // Group 2: 插件工具
        { id: 'controlnet',    icon: '🕹️',  label: 'ControlNet',          type: 'accordion', match: 'ControlNet Integrated' },
        { id: 'proportion',    icon: '🧍',  label: '人体比例',            type: 'accordion', match: '人体头身比例' },
        { id: 'seed',          icon: '🎲',  label: '种子',                type: 'container', elemIds: ['txt2img_seed_row', 'img2img_seed_row'] },
        { id: 'style_presets', icon: '🎨',  label: '关键词预设',          type: 'container', elemIds: ['txt2img_tools', 'txt2img_styles_row', 'img2img_tools', 'img2img_styles_row'] },
        { id: 'sampler',       icon: '⚙️',  label: '采样器',              type: 'container', elemIds: ['sampler_selection_txt2img', 'sampler_selection_img2img', 'txt2img_cfg_scale', 'txt2img_distilled_cfg_scale', 'img2img_cfg_scale', 'img2img_distilled_cfg_scale'] },
        { id: 'model_keyword', icon: '🔑',  label: '模型关键词',          type: 'accordion', match: '模型关键词' },
        { id: 'scripts',       icon: '📜',  label: '脚本',                type: 'script_dropdown' },
        { id: 'adetailer',     icon: '😊',  label: '修脸',                type: 'accordion', match: 'ADetailer' },
        { id: 'regional',      icon: '⊞',  label: '区域控制',            type: 'accordion', match: 'Region' },
        { id: 'see_through',   icon: '✂️',  label: '图层分离',            type: 'accordion', match: 'See-Through 图层分离为PSD文件' },
        { id: 'image_stitch',  icon: '🖼️',  label: '多图拼接',            type: 'accordion', match: '多图拼接参考' },
    ];

    // All tab IDs to hide from the main tab bar
    const HIDDEN_TAB_IDS = [
        'txt2img', 'img2img',
        'pnginfo', 'modelmerger',
        'settings', 'extensions',
        'aesthetic_enhancement_tab', 'Vision_Chat_Tab', 'lighting_assistant',
        'infinite-image-browsing', 'camera_angle_selector', 'sd-webui-image-comparison',
        'model-downloader', 'multimodal_media_tab', 'Segmentation_Tab',
        'trellis2_3d_generator', 'tagger',
        'civitai_interface_neo', 'supermerger', 'sddp-wildcard-manager',
        'extras', 'tutorial_center'
    ];

    // Extra networks sub-tab IDs to hide (nested within txt2img/img2img panels)
    const HIDDEN_EXTRA_TAB_IDS = [
        'txt2img_textual_inversion', 'img2img_textual_inversion',
        'txt2img_checkpoints', 'img2img_checkpoints',
        'txt2img_lora', 'img2img_lora'
    ];

    // --- Storage keys ---
    const STORAGE_KEY_TOPBAR = 'neo_sidebar_topbar_order';
    const STORAGE_KEY_SIDEBAR = 'neo_sidebar_sidebar_order';
    const STORAGE_KEY_VISIBLE_ITEMS = 'neo_sidebar_visible_items';

    // --- Drag & Drop state ---
    let dragSourceId = null;

    // ============================================================
    // Persist item order via localStorage
    // ============================================================
    function loadSavedOrder(key, defaultItems) {
        try {
            const saved = localStorage.getItem(key);
            if (!saved) return null;
            const order = JSON.parse(saved);
            if (!Array.isArray(order) || order.length === 0) return null;

            // Reorder defaultItems based on saved order, appending any new items not in saved order
            const ordered = [];
            const idSet = new Set(defaultItems.map(i => i.id));
            for (const id of order) {
                const item = defaultItems.find(i => i.id === id);
                if (item) {
                    ordered.push(item);
                }
            }
            // Append items that are in defaultItems but not in saved order
            for (const item of defaultItems) {
                if (!order.includes(item.id)) {
                    ordered.push(item);
                }
            }
            return ordered;
        } catch (e) {
            return null;
        }
    }

    function saveOrder(key, items) {
        try {
            const order = items.map(i => i.id);
            localStorage.setItem(key, JSON.stringify(order));
        } catch (e) {
            // localStorage may be unavailable
        }
    }

    function applySavedOrder() {
        const savedTopbar = loadSavedOrder(STORAGE_KEY_TOPBAR, TOPBAR_ITEMS);
        if (savedTopbar) {
            TOPBAR_ITEMS.length = 0;
            TOPBAR_ITEMS.push(...savedTopbar);
        }
        const savedSidebar = loadSavedOrder(STORAGE_KEY_SIDEBAR, SIDEBAR_ITEMS);
        if (savedSidebar) {
            SIDEBAR_ITEMS.length = 0;
            SIDEBAR_ITEMS.push(...savedSidebar);
        }
    }

    // ============================================================
    // Persist item visibility state (which items are shown on main page)
    // ============================================================
    function loadVisibleItems() {
        try {
            const saved = localStorage.getItem(STORAGE_KEY_VISIBLE_ITEMS);
            if (!saved) return null;
            const data = JSON.parse(saved);
            if (typeof data !== 'object') return null;
            return data;
        } catch (e) {
            return null;
        }
    }

    function saveVisibleItems(state) {
        try {
            localStorage.setItem(STORAGE_KEY_VISIBLE_ITEMS, JSON.stringify(state));
        } catch (e) {}
    }

    function isItemVisibleFromState(itemId) {
        const state = loadVisibleItems();
        if (!state) return null;
        return state[itemId] === true ? true : (state[itemId] === false ? false : null);
    }

    function setItemVisibleState(itemId, visible) {
        const state = loadVisibleItems() || {};
        state[itemId] = visible;
        saveVisibleItems(state);
    }

    // ============================================================
    // Check if a plugin/item is actually installed in the UI
    // ============================================================
    function isItemInstalled(item) {
        if (item.type === 'separator') return true;

        if (item.type === 'tab') {
            return document.getElementById(`tab_${item.tabId}-button`) !== null;
        }

        if (item.type === 'accordion') {
            return findAccordionContainers(item.match).length > 0;
        }

        if (item.type === 'script_dropdown') {
            return findScriptDropdownBlocks().length > 0;
        }

        if (item.type === 'container') {
            if (item.elemIds && item.elemIds.length > 0) {
                return item.elemIds.some(function (id) { return document.getElementById(id); });
            }
            return true;
        }

        return true; // unknown type, show it
    }

    // ============================================================
    // Drag & Drop handlers
    // ============================================================
    function onDragStart(e, itemId) {
        dragSourceId = itemId;
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', itemId);
        // Add dragging class after a brief delay
        requestAnimationFrame(() => {
            const btn = e.currentTarget;
            if (btn) btn.classList.add('neo-dragging');
        });
    }

    function onDragEnd(e) {
        document.querySelectorAll('.neo-dragging, .neo-drag-over').forEach(el => {
            el.classList.remove('neo-dragging', 'neo-drag-over');
        });
        dragSourceId = null;
    }

    // Container-level: find the button under the cursor and highlight it
    function onContainerDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';

        // Find the closest button element
        const btn = e.target.closest('[data-sidebar-id]');
        if (!btn) return;

        const targetId = btn.getAttribute('data-sidebar-id');
        if (targetId === dragSourceId) {
            // Don't highlight the source itself
            btn.classList.remove('neo-drag-over');
            return;
        }

        // Remove highlight from all other buttons, add to this one
        document.querySelectorAll('.neo-drag-over').forEach(el => el.classList.remove('neo-drag-over'));
        btn.classList.add('neo-drag-over');
    }

    // Container-level: perform the reorder on drop
    function onContainerDrop(e, containerType) {
        e.preventDefault();
        document.querySelectorAll('.neo-drag-over').forEach(el => el.classList.remove('neo-drag-over'));

        if (!dragSourceId) return;

        const btn = e.target.closest('[data-sidebar-id]');
        if (!btn) return;
        const targetId = btn.getAttribute('data-sidebar-id');
        if (targetId === dragSourceId) return;

        // Determine which array to reorder
        let items, storageKey;
        if (containerType === 'topbar') {
            items = TOPBAR_ITEMS;
            storageKey = STORAGE_KEY_TOPBAR;
        } else {
            items = SIDEBAR_ITEMS;
            storageKey = STORAGE_KEY_SIDEBAR;
        }

        const sourceIdx = items.findIndex(i => i.id === dragSourceId);
        const targetIdx = items.findIndex(i => i.id === targetId);
        if (sourceIdx === -1 || targetIdx === -1) return;

        // Reorder
        const [moved] = items.splice(sourceIdx, 1);
        items.splice(targetIdx, 0, moved);

        // Save order
        saveOrder(storageKey, items);

        // Rebuild the UI
        if (containerType === 'topbar') {
            rebuildTopbar();
        } else {
            rebuildSidebar();
        }
    }

    // Prevent browser default drag behavior (e.g., opening links)
    document.addEventListener('dragover', function (e) { e.preventDefault(); });
    document.addEventListener('drop', function (e) { e.preventDefault(); });

    // ============================================================
    // Rebuild topbar after reorder
    // ============================================================
    function rebuildTopbar() {
        if (!topbarElement) return;
        topbarElement.innerHTML = '';
        for (const item of TOPBAR_ITEMS) {
            if (item.type === 'separator') {
                const sep = document.createElement('div');
                sep.className = 'neo-topbar-sep';
                topbarElement.appendChild(sep);
                continue;
            }
            if (!isItemInstalled(item)) {
                console.log(`[NEO] ✗ Skipping topbar item "${item.label}" (not installed)`);
                continue;
            }
            // Check visibility state: show by default, hide only if explicitly saved as false
            const visibleState = isItemVisibleFromState(item.id);
            if (visibleState === false) {
                continue;
            }
            const btn = createToolbarButton(item, 'topbar');
            topbarElement.appendChild(btn);
        }
    }

    // ============================================================
    // Rebuild sidebar after reorder
    // ============================================================
    function rebuildSidebar() {
        if (!sidebarElement) return;
        sidebarElement.innerHTML = '';
        for (const item of SIDEBAR_ITEMS) {
            if (item.type === 'separator') {
                const sep = document.createElement('div');
                sep.className = 'neo-sidebar-sep';
                sidebarElement.appendChild(sep);
                continue;
            }
            if (!isItemInstalled(item)) {
                console.log(`[NEO] ✗ Skipping sidebar item "${item.label}" (not installed)`);
                continue;
            }
            const btn = createToolbarButton(item, 'sidebar');
            sidebarElement.appendChild(btn);
        }
    }

    // ============================================================
    // Create a single toolbar button with drag support
    // ============================================================
    function createToolbarButton(item, containerType) {
        const isTopbar = containerType === 'topbar';
        const btn = document.createElement('button');
        btn.className = isTopbar ? 'neo-topbar-btn' : 'neo-sidebar-btn';
        btn.setAttribute('data-sidebar-id', item.id);
        btn.setAttribute('title', item.label);
        btn.setAttribute('draggable', 'true');
        btn.innerHTML = `<span class="btn-icon">${item.icon}</span><span class="btn-label">${item.label}</span>`;
        btn.addEventListener('click', handleButtonClick(item));

        // Only dragstart/dragend on the button itself; dragover/drop handled at container level
        btn.addEventListener('dragstart', function (e) {
            onDragStart(e, item.id);
        });
        btn.addEventListener('dragend', onDragEnd);

        return btn;
    }

    // --- State ---
    let topbarElement = null;
    let sidebarElement = null;
    let activeButtonId = null;
    let previousTab = 'txt2img';
    let initialized = false;
    const accordionContainers = {};
    const containerElements = {};
    const scriptDropdownElements = {};

    // ============================================================
    // Find accordion .block elements by label text
    // ============================================================
    function findAccordionContainers(labelText) {
        const containers = [];
        const found = new Set();

        const searchAreas = [
            document.getElementById('tab_txt2img'),
            document.getElementById('tab_img2img'),
            document.querySelector('#tabs'),
            document.getElementById('txt2img_script_container'),
            document.getElementById('img2img_script_container'),
        ];

        for (const area of searchAreas) {
            if (!area) continue;
            const labelWraps = area.querySelectorAll('.label-wrap');
            for (const lw of labelWraps) {
                const labelTextEl = lw.querySelector('.label-text');
                const text = (labelTextEl || lw).textContent.trim();
                if (text.includes(labelText)) {
                    const block = lw.closest('.block');
                    if (block && !found.has(block)) {
                        found.add(block);
                        containers.push(block);
                    }
                }
            }
        }

        // Fallback: search entire document
        if (containers.length === 0) {
            const allLabels = document.querySelectorAll('.label-wrap, .label-text');
            for (const el of allLabels) {
                if (el.closest('.neo-sidebar-hidden-panel')) continue;
                if (el.closest('#neo-sidebar')) continue;
                if (el.closest('#neo-topbar')) continue;
                const text = el.textContent.trim();
                if (text === labelText || text.includes(labelText)) {
                    const block = el.closest('.block');
                    if (block && !found.has(block) && !block.closest('.neo-sidebar-hidden-panel')) {
                        if (block.closest('#tabs') || block.closest('.tabitem')) {
                            found.add(block);
                            containers.push(block);
                            console.log(`[NEO] ✓ Found accordion "${labelText}" via fallback`);
                        }
                    }
                }
            }
        }

        return containers;
    }

    // ============================================================
    // Find Script dropdown .block elements
    // ============================================================
    function findScriptDropdownBlocks() {
        const elements = [];
        const found = new Set();
        const tabIds = ['tab_txt2img', 'tab_img2img'];

        for (const tabId of tabIds) {
            const tab = document.getElementById(tabId);
            if (!tab) continue;
            const dropdown = tab.querySelector('#script_list');
            if (dropdown) {
                const block = dropdown.closest('.block');
                if (block && !found.has(block)) {
                    found.add(block);
                    elements.push(block);
                }
            }
        }
        return elements;
    }

    // ============================================================
    // Find container elements by ID
    // ============================================================
    function findContainerElements(elemIds) {
        const elements = [];
        for (const id of elemIds) {
            const el = document.getElementById(id);
            if (el) elements.push(el);
        }
        return elements;
    }

    // ============================================================
    // Hide all panels initially
    // ============================================================
    function hideAllPanels() {
        // Load saved visible state to restore previously shown items
        const savedVisibleState = loadVisibleItems();

        // Hide sidebar items
        for (const item of SIDEBAR_ITEMS) {
            if (!isItemInstalled(item)) continue;

            // Check if this item was previously visible (shown on main page)
            const wasVisible = savedVisibleState && savedVisibleState[item.id] === true;

            if (item.type === 'accordion') {
                const containers = findAccordionContainers(item.match);
                if (containers.length > 0) {
                    accordionContainers[item.id] = containers;
                    if (wasVisible) {
                        // Show the item (restore saved state)
                        for (const c of containers) {
                            c.classList.remove('neo-sidebar-hidden-panel');
                            c.style.display = '';
                            let content = c.querySelector('.block-content');
                            if (!content) {
                                const labelWrap = c.querySelector('.label-wrap');
                                if (labelWrap) {
                                    labelWrap.setAttribute('aria-expanded', 'true');
                                }
                            } else {
                                content.style.display = '';
                                const labelWrap = c.querySelector('.label-wrap');
                                if (labelWrap) {
                                    labelWrap.setAttribute('aria-expanded', 'true');
                                }
                            }
                        }
                        console.log(`[NEO] ✓ Restored visible accordion "${item.label}"`);
                    } else {
                        // Hide the item
                        for (const c of containers) {
                            c.style.display = 'none';
                            c.classList.add('neo-sidebar-hidden-panel');
                        }
                        console.log(`[NEO] ✓ Hidden accordion "${item.label}"`);
                    }
                } else {
                    console.warn(`[NEO] ✗ Accordion NOT found: "${item.label}"`);
                }
            } else if (item.type === 'container') {
                const elements = findContainerElements(item.elemIds);
                if (elements.length > 0) {
                    containerElements[item.id] = elements;
                    if (wasVisible) {
                        // Show the item
                        for (const el of elements) {
                            el.classList.remove('neo-sidebar-hidden-panel');
                            el.style.display = '';
                        }
                        console.log(`[NEO] ✓ Restored visible container "${item.label}"`);
                    } else {
                        // Hide the item
                        for (const el of elements) {
                            el.style.display = 'none';
                            el.classList.add('neo-sidebar-hidden-panel');
                        }
                        console.log(`[NEO] ✓ Hidden container "${item.label}"`);
                    }
                } else {
                    console.warn(`[NEO] ✗ Container NOT found: "${item.label}"`);
                }
            } else if (item.type === 'model_components') {
                const modelIds = ['forge_ui_preset', 'setting_sd_model_checkpoint', 'setting_sd_modules', 'forge_ui_dtype'];
                if (wasVisible === 'preset') {
                    // Show only UI Preset
                    for (const id of modelIds) {
                        const el = document.getElementById(id);
                        if (el) {
                            if (id === 'forge_ui_preset') {
                                el.classList.remove('neo-sidebar-hidden-panel');
                                el.style.display = '';
                            } else {
                                el.classList.add('neo-sidebar-hidden-panel');
                                el.style.display = 'none';
                            }
                        }
                    }
                    console.log(`[NEO] ✓ Restored model_components preset "${item.label}"`);
                } else if (wasVisible === 'all') {
                    // Show all model elements
                    for (const id of modelIds) {
                        const el = document.getElementById(id);
                        if (el) {
                            el.classList.remove('neo-sidebar-hidden-panel');
                            el.style.display = '';
                        }
                    }
                    console.log(`[NEO] ✓ Restored model_components all "${item.label}"`);
                } else {
                    // Hide all model elements
                    for (const id of modelIds) {
                        const el = document.getElementById(id);
                        if (el) {
                            el.style.display = 'none';
                            el.classList.add('neo-sidebar-hidden-panel');
                        }
                    }
                    console.log(`[NEO] ✓ Hidden model_components "${item.label}"`);
                }
            } else if (item.type === 'script_dropdown') {
                const blocks = findScriptDropdownBlocks();
                if (blocks.length > 0) {
                    scriptDropdownElements[item.id] = blocks;
                    if (wasVisible) {
                        // Show the item
                        for (const b of blocks) {
                            b.classList.remove('neo-sidebar-hidden-panel');
                            b.style.display = '';
                        }
                        console.log(`[NEO] ✓ Restored visible script dropdown`);
                    } else {
                        // Hide the item
                        for (const b of blocks) {
                            b.style.display = 'none';
                            b.classList.add('neo-sidebar-hidden-panel');
                        }
                        console.log(`[NEO] ✓ Hidden script dropdown (${blocks.length} instances)`);
                    }
                } else {
                    console.warn(`[NEO] ✗ Script dropdown NOT found`);
                }
            }
        }
        // Hide top toolbar accordion items
        for (const item of TOPBAR_ITEMS) {
            if (!isItemInstalled(item)) continue;

            // Check if this item was previously visible
            const wasVisible = savedVisibleState && savedVisibleState[item.id] === true;

            if (item.type === 'accordion') {
                const containers = findAccordionContainers(item.match);
                if (containers.length > 0) {
                    accordionContainers[item.id] = containers;
                    if (wasVisible) {
                        // Show the item
                        for (const c of containers) {
                            c.classList.remove('neo-sidebar-hidden-panel');
                            c.style.display = '';
                            let content = c.querySelector('.block-content');
                            if (!content) {
                                const labelWrap = c.querySelector('.label-wrap');
                                if (labelWrap) {
                                    labelWrap.setAttribute('aria-expanded', 'true');
                                }
                            } else {
                                content.style.display = '';
                                const labelWrap = c.querySelector('.label-wrap');
                                if (labelWrap) {
                                    labelWrap.setAttribute('aria-expanded', 'true');
                                }
                            }
                        }
                        console.log(`[NEO] ✓ Restored visible accordion "${item.label}"`);
                    } else {
                        // Hide the item
                        for (const c of containers) {
                            c.style.display = 'none';
                            c.classList.add('neo-sidebar-hidden-panel');
                        }
                        console.log(`[NEO] ✓ Hidden accordion "${item.label}"`);
                    }
                } else {
                    console.warn(`[NEO] ✗ Accordion NOT found: "${item.label}"`);
                }
            }
        }

        // After restoring visible items, update active button state
        if (savedVisibleState) {
            for (const item of SIDEBAR_ITEMS) {
                if (savedVisibleState[item.id] === true) {
                    setActiveButton(item.id);
                }
            }
        }
    }

    // ============================================================
    // Toggle accordion panel (show/hide)
    // ============================================================
    function toggleAccordionPanel(item) {
        let containers = accordionContainers[item.id];
        if (!containers || containers.length === 0) {
            containers = findAccordionContainers(item.match);
            if (containers.length === 0) {
                console.warn('[NEO] Cannot find accordion:', item.label);
                return false;
            }
            accordionContainers[item.id] = containers;
        }

        const isHidden = containers[0].classList.contains('neo-sidebar-hidden-panel');

        if (isHidden) {
            for (const c of containers) {
                c.classList.remove('neo-sidebar-hidden-panel');
                c.style.display = '';

                let content = c.querySelector('.block-content');
                if (!content) {
                    const labelWrap = c.querySelector('.label-wrap');
                    if (labelWrap) {
                        labelWrap.dispatchEvent(new MouseEvent('click', {
                            bubbles: true,
                            cancelable: true,
                            view: window
                        }));
                    }
                } else {
                    content.style.display = '';
                    const labelWrap = c.querySelector('.label-wrap');
                    if (labelWrap) {
                        labelWrap.setAttribute('aria-expanded', 'true');
                    }
                }
            }
            setActiveButton(item.id);
            setItemVisibleState(item.id, true);
        } else {
            for (const c of containers) {
                c.classList.add('neo-sidebar-hidden-panel');
                c.style.display = 'none';
            }
            clearActiveButton();
            setItemVisibleState(item.id, false);
        }
        return true;
    }

    // ============================================================
    // Toggle container panel (show/hide)
    // ============================================================
    function toggleContainerPanel(item) {
        let elements = containerElements[item.id];
        if (!elements || elements.length === 0) {
            elements = findContainerElements(item.elemIds);
            if (elements.length === 0) return false;
            containerElements[item.id] = elements;
        }

        const isHidden = elements[0].classList.contains('neo-sidebar-hidden-panel');

        if (isHidden) {
            for (const el of elements) {
                el.classList.remove('neo-sidebar-hidden-panel');
                el.style.display = '';
            }
            setActiveButton(item.id);
            setItemVisibleState(item.id, true);
        } else {
            for (const el of elements) {
                el.classList.add('neo-sidebar-hidden-panel');
                el.style.display = 'none';
            }
            clearActiveButton();
            setItemVisibleState(item.id, false);
        }
        return true;
    }

    // ============================================================
    // Toggle Model Components (hidden → UI Preset only, then toggle All ↔ Preset)
    // ============================================================
    function toggleModelComponents(item) {
        const uiPreset = document.getElementById('forge_ui_preset');
        if (!uiPreset) return;

        const isHidden = uiPreset.classList.contains('neo-sidebar-hidden-panel');
        const modelIds = ['setting_sd_model_checkpoint', 'setting_sd_modules', 'forge_ui_dtype'];

        // Check if model elements are currently visible
        let modelVisible = false;
        for (const id of modelIds) {
            const el = document.getElementById(id);
            if (el && !el.classList.contains('neo-sidebar-hidden-panel') && el.style.display !== 'none') {
                modelVisible = true;
                break;
            }
        }

        if (isHidden) {
            // Click 1: Show UI Preset only (first expansion)
            uiPreset.classList.remove('neo-sidebar-hidden-panel');
            uiPreset.style.display = '';
            for (const id of modelIds) {
                const el = document.getElementById(id);
                if (el) {
                    el.classList.add('neo-sidebar-hidden-panel');
                    el.style.display = 'none';
                }
            }
            setActiveButton(item.id);
            setItemVisibleState(item.id, 'preset');
        } else {
            // Toggle between 'all' and 'preset' only (no more hide state)
            if (modelVisible) {
                // All visible -> back to UI Preset only
                for (const id of modelIds) {
                    const el = document.getElementById(id);
                    if (el) {
                        el.classList.add('neo-sidebar-hidden-panel');
                        el.style.display = 'none';
                    }
                }
                setItemVisibleState(item.id, 'preset');
            } else {
                // UI Preset only -> show all
                for (const id of modelIds) {
                    const el = document.getElementById(id);
                    if (el) {
                        el.classList.remove('neo-sidebar-hidden-panel');
                        el.style.display = '';
                    }
                }
                setItemVisibleState(item.id, 'all');
            }
            setActiveButton(item.id);
        }
    }

    // ============================================================
    // Toggle Script dropdown (show/hide)
    // ============================================================
    function toggleScriptDropdown(item) {
        let blocks = scriptDropdownElements[item.id];
        if (!blocks || blocks.length === 0) {
            blocks = findScriptDropdownBlocks();
            if (blocks.length === 0) return false;
            scriptDropdownElements[item.id] = blocks;
        }

        const isHidden = blocks[0].classList.contains('neo-sidebar-hidden-panel');

        if (isHidden) {
            for (const b of blocks) {
                b.classList.remove('neo-sidebar-hidden-panel');
                b.style.display = '';
            }
            setActiveButton(item.id);
            setItemVisibleState(item.id, true);
        } else {
            for (const b of blocks) {
                b.classList.add('neo-sidebar-hidden-panel');
                b.style.display = 'none';
            }
            clearActiveButton();
            setItemVisibleState(item.id, false);
        }
        return true;
    }

    // ============================================================
    // Tab switching (with toggle)
    // ============================================================
    function switchToTab(tabId) {
        const targetBtn = document.getElementById(`tab_${tabId}-button`);
        if (!targetBtn) return false;

        const isActive = targetBtn.classList.contains('selected') || targetBtn.getAttribute('aria-selected') === 'true';

        if (isActive) {
            // Toggle off: switch back to the previous tab
            const prevBtn = document.getElementById(`tab_${previousTab}-button`);
            if (prevBtn) {
                prevBtn.click();
                clearActiveButton();
                return true;
            }
            return false;
        }

        // Save previous tab BEFORE switching
        const txt2imgBtn = document.getElementById('tab_txt2img-button');
        const img2imgBtn = document.getElementById('tab_img2img-button');
        if (txt2imgBtn && txt2imgBtn.classList.contains('selected')) {
            previousTab = 'txt2img';
        } else if (img2imgBtn && img2imgBtn.classList.contains('selected')) {
            previousTab = 'img2img';
        }

        targetBtn.click();
        return true;
    }

    // ============================================================
    // Switch between txt2img and img2img (generation tab)
    // ============================================================
    function switchGenerationTab() {
        const txt2imgBtn = document.getElementById('tab_txt2img-button');
        const img2imgBtn = document.getElementById('tab_img2img-button');
        if (!txt2imgBtn && !img2imgBtn) return;

        // Simply toggle between txt2img and img2img
        const onTxt2img = txt2imgBtn && (txt2imgBtn.classList.contains('selected') || txt2imgBtn.getAttribute('aria-selected') === 'true');
        const onImg2img = img2imgBtn && (img2imgBtn.classList.contains('selected') || img2imgBtn.getAttribute('aria-selected') === 'true');

        if (onTxt2img && img2imgBtn) {
            img2imgBtn.click();
        } else if (onImg2img && txt2imgBtn) {
            txt2imgBtn.click();
        } else if (txt2imgBtn) {
            txt2imgBtn.click();
        } else if (img2imgBtn) {
            img2imgBtn.click();
        }
    }

    // ============================================================
    // Extra networks tab switching (模型, Lora, 嵌入式)
    // ============================================================
    function switchToExtraTab(item) {
        const isImg2img = document.getElementById('img2img_tab_selected') !== null ||
            document.getElementById('tab_img2img-button')?.getAttribute('aria-selected') === 'true' ||
            document.getElementById('tab_img2img-button')?.classList.contains('selected');

        const tabId = isImg2img ? item.tabIds[1] : item.tabIds[0];
        const btn = document.getElementById(tabId + '-button');

        if (!btn) return;

        const mainTabId = isImg2img ? 'img2img' : 'txt2img';

        // Check if the extra tab is already active
        const isActive = btn.classList.contains('selected') || btn.getAttribute('aria-selected') === 'true';
        if (isActive) {
            // Toggle off: switch to the main generation tab view
            const mainGenBtn = document.getElementById(`tab_${mainTabId}-button`);
            if (mainGenBtn) {
                mainGenBtn.click();
            }
            clearActiveButton();
            return;
        }

        // First switch to the generation tab, then show the extra sub-tab
        const mainGenBtn = document.getElementById(`tab_${mainTabId}-button`);
        if (mainGenBtn) {
            mainGenBtn.click();
        }
        // Small delay to let Gradio switch tabs, then click the extra sub-tab
        setTimeout(function () {
            btn.click();
        }, 50);
        setActiveButton(item.id);
    }

    // ============================================================
    // Sub-tab switching (e.g., 多媒体处理的子标签页)
    // ============================================================
    function switchToSubTab(item) {
        const mainTabId = item.tabId;
        const subTabLabel = item.subtabLabel;

        // First switch to the main tab
        const mainBtn = document.getElementById(`tab_${mainTabId}-button`);
        if (!mainBtn) return;

        const isAlreadyOnMain = mainBtn.classList.contains('selected') || mainBtn.getAttribute('aria-selected') === 'true';

        if (!isAlreadyOnMain) {
            mainBtn.click();
        }

        // Find and click the sub-tab button by its text content
        setTimeout(function() {
            const buttons = document.querySelectorAll('button');
            for (let i = 0; i < buttons.length; i++) {
                const btn = buttons[i];
                const role = btn.getAttribute('role') || '';
                if (role !== 'tab' && role !== 'tab-nav') continue;
                if (btn.textContent.trim() === subTabLabel) {
                    btn.click();
                    return;
                }
            }
            // Fallback: try fuzzy match
            for (let i = 0; i < buttons.length; i++) {
                const btn = buttons[i];
                const role = btn.getAttribute('role') || '';
                if (role !== 'tab' && role !== 'tab-nav') continue;
                if (btn.textContent.trim().includes(subTabLabel) || subTabLabel.includes(btn.textContent.trim())) {
                    btn.click();
                    return;
                }
            }
        }, 100);

        setActiveButton(item.id);
    }

    // ============================================================
    // Button state management
    // ============================================================
    function getContainer() {
        // Return the container that holds the active button (topbar or sidebar)
        if (activeButtonId) {
            if (topbarElement && topbarElement.querySelector(`[data-sidebar-id="${activeButtonId}"]`)) {
                return topbarElement;
            }
            if (sidebarElement && sidebarElement.querySelector(`[data-sidebar-id="${activeButtonId}"]`)) {
                return sidebarElement;
            }
        }
        return null;
    }

    function setActiveButton(id) {
        // Clear previous active button in both containers
        if (activeButtonId) {
            const prevContainer = getContainer();
            if (prevContainer) {
                const prev = prevContainer.querySelector(`[data-sidebar-id="${activeButtonId}"]`);
                if (prev) prev.classList.remove('active');
            }
        }
        activeButtonId = id;
        // Apply active to new button (check both containers)
        if (topbarElement) {
            const btn = topbarElement.querySelector(`[data-sidebar-id="${id}"]`);
            if (btn) { btn.classList.add('active'); return; }
        }
        if (sidebarElement) {
            const btn = sidebarElement.querySelector(`[data-sidebar-id="${id}"]`);
            if (btn) { btn.classList.add('active'); return; }
        }
    }

    function clearActiveButton() {
        if (activeButtonId) {
            const container = getContainer();
            if (container) {
                const prev = container.querySelector(`[data-sidebar-id="${activeButtonId}"]`);
                if (prev) prev.classList.remove('active');
            }
        }
        activeButtonId = null;
    }

    // ============================================================
    // Button click handler
    // ============================================================
    function handleButtonClick(item) {
        return function () {
            if (item.type === 'tab') {
                switchToTab(item.tabId);
                setActiveButton(item.id);
            } else if (item.type === 'generation_tab') {
                switchGenerationTab();
            } else if (item.type === 'accordion') {
                toggleAccordionPanel(item);
            } else if (item.type === 'container') {
                toggleContainerPanel(item);
            } else if (item.type === 'script_dropdown') {
                toggleScriptDropdown(item);
            } else if (item.type === 'extra_tab') {
                switchToExtraTab(item);
            } else if (item.type === 'subtab') {
                switchToSubTab(item);
            } else if (item.type === 'model_components') {
                toggleModelComponents(item);
            }
        };
    }

    // ============================================================
    // Create top toolbar DOM
    // ============================================================
    function createTopbar() {
        if (document.getElementById('neo-topbar')) return;

        // Retry if not all tab-type items have their buttons yet (Gradio creates them async)
        const tabItems = TOPBAR_ITEMS.filter(function (item) {
            return item.type === 'tab' || item.type === 'subtab' || item.type === 'extra_tab';
        });
        const allTabsReady = tabItems.every(function (item) { return isItemInstalled(item); });
        if (!allTabsReady) {
            if (!createTopbar._retryCount) createTopbar._retryCount = 0;
            createTopbar._retryCount++;
            if (createTopbar._retryCount <= 20) {
                setTimeout(createTopbar, 500);
                return;
            }
        }

        topbarElement = document.createElement('div');
        topbarElement.id = 'neo-topbar';

        // Container-level drag handlers
        topbarElement.addEventListener('dragover', onContainerDragOver);
        topbarElement.addEventListener('drop', function (e) { onContainerDrop(e, 'topbar'); });
        topbarElement.addEventListener('dragleave', function (e) {
            // Remove highlight when leaving the container
            document.querySelectorAll('.neo-drag-over').forEach(el => el.classList.remove('neo-drag-over'));
        });

        for (const item of TOPBAR_ITEMS) {
            if (item.type === 'separator') {
                const sep = document.createElement('div');
                sep.className = 'neo-topbar-sep';
                topbarElement.appendChild(sep);
                continue;
            }
            if (!isItemInstalled(item)) {
                console.log(`[NEO] ✗ Skipping topbar item "${item.label}" (not installed)`);
                continue;
            }
            // Check visibility state: show by default, hide only if explicitly saved as false
            const visibleState = isItemVisibleFromState(item.id);
            if (visibleState === false) {
                continue;
            }
            const btn = createToolbarButton(item, 'topbar');
            topbarElement.appendChild(btn);
        }

        document.body.appendChild(topbarElement);
    }

    // ============================================================
    // Create sidebar DOM
    // ============================================================
    function createSidebar() {
        if (document.getElementById('neo-sidebar')) return;

        sidebarElement = document.createElement('div');
        sidebarElement.id = 'neo-sidebar';

        // Container-level drag handlers
        sidebarElement.addEventListener('dragover', onContainerDragOver);
        sidebarElement.addEventListener('drop', function (e) { onContainerDrop(e, 'sidebar'); });
        sidebarElement.addEventListener('dragleave', function (e) {
            document.querySelectorAll('.neo-drag-over').forEach(el => el.classList.remove('neo-drag-over'));
        });

        for (const item of SIDEBAR_ITEMS) {
            if (item.type === 'separator') {
                const sep = document.createElement('div');
                sep.className = 'neo-sidebar-sep';
                sidebarElement.appendChild(sep);
                continue;
            }
            if (!isItemInstalled(item)) {
                console.log(`[NEO] ✗ Skipping sidebar item "${item.label}" (not installed)`);
                continue;
            }
            const btn = createToolbarButton(item, 'sidebar');
            sidebarElement.appendChild(btn);
        }

        // 配置按钮（最后一个）
        const configSep = document.createElement('div');
        configSep.className = 'neo-sidebar-sep';
        sidebarElement.appendChild(configSep);

        const configBtn = document.createElement('button');
        configBtn.className = 'neo-sidebar-btn neo-sidebar-config-btn';
        configBtn.setAttribute('title', '配置预设 - 选择哪些项目显示在主页面');
        configBtn.innerHTML = '<span class="btn-icon">⚙️</span><span class="btn-label">配置</span>';
        configBtn.addEventListener('click', showSettingsDialog);
        sidebarElement.appendChild(configBtn);

        document.body.appendChild(sidebarElement);
    }

    // ============================================================
    // Show settings dialog (preset configuration)
    // ============================================================
    function showSettingsDialog() {
        // Remove existing dialog if any
        const existing = document.getElementById('neo-settings-dialog');
        if (existing) existing.remove();

        const overlay = document.createElement('div');
        overlay.id = 'neo-settings-dialog';
        overlay.className = 'neo-settings-overlay';

        const dialog = document.createElement('div');
        dialog.className = 'neo-settings-dialog';

        // Header
        const header = document.createElement('div');
        header.className = 'neo-settings-header';
        header.innerHTML = '<span class="neo-settings-title">⚙️ 侧边栏预设配置</span>';
        const closeBtn = document.createElement('button');
        closeBtn.className = 'neo-settings-close';
        closeBtn.innerHTML = '✕';
        closeBtn.addEventListener('click', function () { overlay.remove(); });
        header.appendChild(closeBtn);
        dialog.appendChild(header);

        // Subtitle
        const subtitle = document.createElement('div');
        subtitle.className = 'neo-settings-subtitle';
        subtitle.textContent = '勾选要显示在主页面上的项目，取消勾选则收起到侧边栏';
        dialog.appendChild(subtitle);

        // Load current state
        const currentState = loadVisibleItems() || {};

        // Content - scrollable list
        const content = document.createElement('div');
        content.className = 'neo-settings-content';

        // Helper to create a section
        function createSection(title, items) {
            const section = document.createElement('div');
            section.className = 'neo-settings-section';

            const sectionTitle = document.createElement('div');
            sectionTitle.className = 'neo-settings-section-title';
            sectionTitle.textContent = title;
            section.appendChild(sectionTitle);

            for (const item of items) {
                if (item.type === 'separator' || !item.id) continue;
                if (!isItemInstalled(item)) continue;

                const row = document.createElement('label');
                row.className = 'neo-settings-row';

                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.className = 'neo-settings-cb';
                cb.dataset.itemId = item.id;
                // Default: checked unless explicitly hidden by saved state
                cb.checked = currentState[item.id] !== false;

                const icon = document.createElement('span');
                icon.className = 'neo-settings-item-icon';
                icon.textContent = item.icon || '▪️';

                const label = document.createElement('span');
                label.className = 'neo-settings-item-label';
                label.textContent = item.label;

                const hint = document.createElement('span');
                hint.className = 'neo-settings-item-hint';
                hint.textContent = cb.checked ? '显示中' : '已收起';

                // Update hint on checkbox change
                cb.addEventListener('change', function () {
                    hint.textContent = this.checked ? '显示中' : '已收起';
                });

                row.appendChild(cb);
                row.appendChild(icon);
                row.appendChild(label);
                row.appendChild(hint);
                section.appendChild(row);
            }

            return section;
        }

        // Topbar section
        content.appendChild(createSection('上边栏', TOPBAR_ITEMS));
        // Sidebar section
        content.appendChild(createSection('侧边栏', SIDEBAR_ITEMS));

        dialog.appendChild(content);

        // Footer with buttons
        const footer = document.createElement('div');
        footer.className = 'neo-settings-footer';

        const resetBtn = document.createElement('button');
        resetBtn.className = 'neo-settings-btn neo-settings-btn-reset';
        resetBtn.textContent = '重置全部收起';
        resetBtn.addEventListener('click', function () {
            // Uncheck all
            const cbs = dialog.querySelectorAll('.neo-settings-cb');
            for (const cb of cbs) {
                cb.checked = false;
            }
            const hints = dialog.querySelectorAll('.neo-settings-item-hint');
            for (const h of hints) {
                h.textContent = '已收起';
            }
        });
        footer.appendChild(resetBtn);

        const saveBtn = document.createElement('button');
        saveBtn.className = 'neo-settings-btn neo-settings-btn-save';
        saveBtn.textContent = '✅ 保存设置';
        saveBtn.addEventListener('click', function () {
            // Collect state
            const newState = {};
            const cbs = dialog.querySelectorAll('.neo-settings-cb');
            for (const cb of cbs) {
                newState[cb.dataset.itemId] = cb.checked;
            }
            // Save to localStorage
            saveVisibleItems(newState);
            // Apply immediately
            applyVisibilitySettings(newState);
            // Close dialog
            overlay.remove();
        });
        footer.appendChild(saveBtn);

        dialog.appendChild(footer);
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);

        // Close on overlay click (but not on dialog click)
        overlay.addEventListener('click', function (e) {
            if (e.target === overlay) overlay.remove();
        });
    }

    // ============================================================
    // Apply visibility settings (show/hide panels based on state)
    // ============================================================
    function applyVisibilitySettings(state) {
        // Helper: show or hide a panel
        function setPanelVisibility(item, visible) {
            if (item.type === 'accordion') {
                let containers = accordionContainers[item.id];
                if (!containers || containers.length === 0) {
                    containers = findAccordionContainers(item.match);
                    if (containers.length === 0) return;
                    accordionContainers[item.id] = containers;
                }
                for (const c of containers) {
                    if (visible) {
                        c.classList.remove('neo-sidebar-hidden-panel');
                        c.style.display = '';
                        let content = c.querySelector('.block-content');
                        if (!content) {
                            const labelWrap = c.querySelector('.label-wrap');
                            if (labelWrap) labelWrap.setAttribute('aria-expanded', 'true');
                        } else {
                            content.style.display = '';
                            const labelWrap = c.querySelector('.label-wrap');
                            if (labelWrap) labelWrap.setAttribute('aria-expanded', 'true');
                        }
                    } else {
                        c.classList.add('neo-sidebar-hidden-panel');
                        c.style.display = 'none';
                    }
                }
            } else if (item.type === 'container') {
                let elements = containerElements[item.id];
                if (!elements || elements.length === 0) {
                    elements = findContainerElements(item.elemIds);
                    if (elements.length === 0) return;
                    containerElements[item.id] = elements;
                }
                for (const el of elements) {
                    if (visible) {
                        el.classList.remove('neo-sidebar-hidden-panel');
                        el.style.display = '';
                    } else {
                        el.classList.add('neo-sidebar-hidden-panel');
                        el.style.display = 'none';
                    }
                }
            } else if (item.type === 'script_dropdown') {
                let blocks = scriptDropdownElements[item.id];
                if (!blocks || blocks.length === 0) {
                    blocks = findScriptDropdownBlocks();
                    if (blocks.length === 0) return;
                    scriptDropdownElements[item.id] = blocks;
                }
                for (const b of blocks) {
                    if (visible) {
                        b.classList.remove('neo-sidebar-hidden-panel');
                        b.style.display = '';
                    } else {
                        b.classList.add('neo-sidebar-hidden-panel');
                        b.style.display = 'none';
                    }
                }
            } else if (item.type === 'tab' || item.type === 'subtab' || item.type === 'extra_tab') {
                // Show/hide the tab panel on the main page (permanently show)
                if (item.type === 'tab') {
                    const panel = document.getElementById('tab_' + item.tabId);
                    if (panel) {
                        if (visible) {
                            panel.style.display = '';
                            panel.classList.remove('neo-sidebar-hidden-panel');
                        } else {
                            panel.style.display = 'none';
                            panel.classList.add('neo-sidebar-hidden-panel');
                        }
                    }
                }
                return;
            }
        }

        // Apply to all items
        const allItems = [...TOPBAR_ITEMS, ...SIDEBAR_ITEMS];
        for (const item of allItems) {
            if (item.type === 'separator' || !item.id) continue;
            if (!isItemInstalled(item)) continue;
            const visible = state[item.id] === true;
            setPanelVisibility(item, visible);
            // Update button active state
            if (visible) {
                setActiveButton(item.id);
            }
        }

        // Rebuild topbar to reflect hidden/shown tab items
        rebuildTopbar();

        console.log('[NEO] ✓ Applied visibility settings');
    }

    // ============================================================
    // Hide non-essential tabs from top-level tab bar
    // ============================================================
    function hideTabs() {
        for (const tabId of HIDDEN_TAB_IDS) {
            const btn = document.getElementById(`tab_${tabId}-button`);
            if (btn) {
                btn.style.display = 'none';
                btn.classList.add('neo-sidebar-hidden-tab');
            } else {
                const tabNav = document.querySelector('#tabs > .tab-nav');
                if (!tabNav) continue;
                const btns = tabNav.querySelectorAll(':scope > button');
                for (const b of btns) {
                    if (b.textContent.trim().toLowerCase().includes(tabId.toLowerCase())) {
                        b.style.display = 'none';
                        b.classList.add('neo-sidebar-hidden-tab');
                        break;
                    }
                }
            }
        }

        // Hide extra networks sub-tabs (nested within txt2img/img2img panels)
        for (const tabId of HIDDEN_EXTRA_TAB_IDS) {
            // Try direct ID lookup (button IDs are like "txt2img_checkpoints-button")
            const btn = document.getElementById(`${tabId}-button`);
            if (btn) {
                btn.style.display = 'none';
                btn.classList.add('neo-sidebar-hidden-tab');
                continue;
            }

            // Search within nested tabs containers
            const extraContainers = ['#txt2img_extra_tabs', '#img2img_extra_tabs'];
            for (const containerSel of extraContainers) {
                const container = document.querySelector(containerSel);
                if (!container) continue;
                const tabNav = container.querySelector('.tab-nav');
                if (!tabNav) continue;
                const btns = tabNav.querySelectorAll(':scope > button');
                for (const b of btns) {
                    if (b.id === `${tabId}-button` || b.textContent.trim().toLowerCase().includes(tabId.toLowerCase())) {
                        b.style.display = 'none';
                        b.classList.add('neo-sidebar-hidden-tab');
                        break;
                    }
                }
            }
        }
    }

    // ============================================================
    // Observe tab changes to sync active state
    // ============================================================
    function setupTabObserver() {
        const allItems = [...TOPBAR_ITEMS, ...SIDEBAR_ITEMS];
        const observer = new MutationObserver(function () {
            // Update left/top buttons
            for (const item of allItems) {
                if (item.type !== 'tab') continue;
                const tabBtn = document.getElementById(`tab_${item.tabId}-button`);
                if (!tabBtn) continue;
                const isActive = tabBtn.classList.contains('selected') ||
                                 tabBtn.getAttribute('aria-selected') === 'true';

                // Update button in both topbar and sidebar
                if (topbarElement) {
                    const btn = topbarElement.querySelector(`[data-sidebar-id="${item.id}"]`);
                    if (btn) btn.classList.toggle('active', isActive);
                }
                if (sidebarElement) {
                    const btn = sidebarElement.querySelector(`[data-sidebar-id="${item.id}"]`);
                    if (btn) btn.classList.toggle('active', isActive);
                }
                if (isActive) activeButtonId = item.id;
            }

            // Update right sidebar tab (txt2img vs img2img)
            const txt2imgTab = document.getElementById('tab_txt2img-button');
            const img2imgTab = document.getElementById('tab_img2img-button');
            if (txt2imgTab && img2imgTab) {
                const isTxt2img = txt2imgTab.classList.contains('selected') ||
                                  txt2imgTab.getAttribute('aria-selected') === 'true';
                showRightbarTab(isTxt2img ? 'txt2img' : 'img2img');
                showBottombarTab(isTxt2img ? 'txt2img' : 'img2img');
            }
        });

        observer.observe(document.body, {
            attributes: true,
            childList: true,
            subtree: true,
            attributeFilter: ['class', 'aria-selected']
        });
    }

    // ============================================================
    // Retry hiding panels
    // ============================================================
    function retryHidePanels() {
        const savedVisibleState = loadVisibleItems();

        for (const item of SIDEBAR_ITEMS) {
            if (!isItemInstalled(item)) continue;

            // Check if this item was previously visible
            const wasVisible = savedVisibleState && savedVisibleState[item.id] === true;

            if (item.type === 'accordion') {
                if (accordionContainers[item.id] && accordionContainers[item.id].length > 0) continue;
                const containers = findAccordionContainers(item.match);
                if (containers.length > 0) {
                    accordionContainers[item.id] = containers;
                    if (wasVisible) {
                        for (const c of containers) {
                            c.classList.remove('neo-sidebar-hidden-panel');
                            c.style.display = '';
                            let content = c.querySelector('.block-content');
                            if (!content) {
                                const labelWrap = c.querySelector('.label-wrap');
                                if (labelWrap) {
                                    labelWrap.setAttribute('aria-expanded', 'true');
                                }
                            } else {
                                content.style.display = '';
                                const labelWrap = c.querySelector('.label-wrap');
                                if (labelWrap) {
                                    labelWrap.setAttribute('aria-expanded', 'true');
                                }
                            }
                        }
                        console.log(`[NEO] ✓ (retry) Restored visible accordion "${item.label}"`);
                    } else {
                        for (const c of containers) {
                            c.style.display = 'none';
                            c.classList.add('neo-sidebar-hidden-panel');
                        }
                        console.log(`[NEO] ✓ (retry) Hidden accordion "${item.label}"`);
                    }
                }
            } else if (item.type === 'container') {
                if (containerElements[item.id] && containerElements[item.id].length > 0) continue;
                const elements = findContainerElements(item.elemIds);
                if (elements.length > 0) {
                    containerElements[item.id] = elements;
                    if (wasVisible) {
                        for (const el of elements) {
                            el.classList.remove('neo-sidebar-hidden-panel');
                            el.style.display = '';
                        }
                        console.log(`[NEO] ✓ (retry) Restored visible container "${item.label}"`);
                    } else {
                        for (const el of elements) {
                            el.style.display = 'none';
                            el.classList.add('neo-sidebar-hidden-panel');
                        }
                        console.log(`[NEO] ✓ (retry) Hidden container "${item.label}"`);
                    }
                }
            } else if (item.type === 'model_components') {
                const modelIds = ['forge_ui_preset', 'setting_sd_model_checkpoint', 'setting_sd_modules', 'forge_ui_dtype'];
                if (wasVisible === 'preset') {
                    for (const id of modelIds) {
                        const el = document.getElementById(id);
                        if (el) {
                            if (id === 'forge_ui_preset') {
                                el.classList.remove('neo-sidebar-hidden-panel');
                                el.style.display = '';
                            } else {
                                el.classList.add('neo-sidebar-hidden-panel');
                                el.style.display = 'none';
                            }
                        }
                    }
                    console.log(`[NEO] ✓ (retry) Restored model_components preset "${item.label}"`);
                } else if (wasVisible === 'all') {
                    for (const id of modelIds) {
                        const el = document.getElementById(id);
                        if (el) {
                            el.classList.remove('neo-sidebar-hidden-panel');
                            el.style.display = '';
                        }
                    }
                    console.log(`[NEO] ✓ (retry) Restored model_components all "${item.label}"`);
                } else {
                    for (const id of modelIds) {
                        const el = document.getElementById(id);
                        if (el) {
                            el.style.display = 'none';
                            el.classList.add('neo-sidebar-hidden-panel');
                        }
                    }
                    console.log(`[NEO] ✓ (retry) Hidden model_components "${item.label}"`);
                }
            } else if (item.type === 'script_dropdown') {
                if (scriptDropdownElements[item.id] && scriptDropdownElements[item.id].length > 0) continue;
                const blocks = findScriptDropdownBlocks();
                if (blocks.length > 0) {
                    scriptDropdownElements[item.id] = blocks;
                    if (wasVisible) {
                        for (const b of blocks) {
                            b.classList.remove('neo-sidebar-hidden-panel');
                            b.style.display = '';
                        }
                        console.log(`[NEO] ✓ (retry) Restored visible script dropdown`);
                    } else {
                        for (const b of blocks) {
                            b.style.display = 'none';
                            b.classList.add('neo-sidebar-hidden-panel');
                        }
                        console.log(`[NEO] ✓ (retry) Hidden script dropdown`);
                    }
                }
            }
        }
        // Retry hiding top toolbar accordion items
        for (const item of TOPBAR_ITEMS) {
            if (!isItemInstalled(item)) continue;

            const wasVisible = savedVisibleState && savedVisibleState[item.id] === true;

            if (item.type === 'accordion') {
                if (accordionContainers[item.id] && accordionContainers[item.id].length > 0) continue;
                const containers = findAccordionContainers(item.match);
                if (containers.length > 0) {
                    accordionContainers[item.id] = containers;
                    if (wasVisible) {
                        for (const c of containers) {
                            c.classList.remove('neo-sidebar-hidden-panel');
                            c.style.display = '';
                            let content = c.querySelector('.block-content');
                            if (!content) {
                                const labelWrap = c.querySelector('.label-wrap');
                                if (labelWrap) {
                                    labelWrap.setAttribute('aria-expanded', 'true');
                                }
                            } else {
                                content.style.display = '';
                                const labelWrap = c.querySelector('.label-wrap');
                                if (labelWrap) {
                                    labelWrap.setAttribute('aria-expanded', 'true');
                                }
                            }
                        }
                        console.log(`[NEO] ✓ (retry) Restored visible accordion "${item.label}"`);
                    } else {
                        for (const c of containers) {
                            c.style.display = 'none';
                            c.classList.add('neo-sidebar-hidden-panel');
                        }
                        console.log(`[NEO] ✓ (retry) Hidden accordion "${item.label}"`);
                    }
                }
            }
        }
    }

    // ============================================================
    // Apply topbar panel visibility state (permanently show/hide tab panels)
    // ============================================================
    function applyTopbarPanelState() {
        const saved = loadVisibleItems();
        if (!saved) return;
        for (const item of TOPBAR_ITEMS) {
            if (item.type !== 'tab' || !item.tabId) continue;
            if (!isItemInstalled(item)) continue;
            const visible = saved[item.id] === true;
            const panel = document.getElementById('tab_' + item.tabId);
            if (panel) {
                if (visible) {
                    panel.style.display = '';
                    panel.classList.remove('neo-sidebar-hidden-panel');
                } else {
                    panel.style.display = 'none';
                    panel.classList.add('neo-sidebar-hidden-panel');
                }
            }
        }
    }

    // ============================================================
    // Create right sidebar for output widgets
    // ============================================================
    function createRightbar() {
        if (document.getElementById('neo-rightbar')) return;

        const rightbar = document.createElement('div');
        rightbar.id = 'neo-rightbar';

        // Will be populated by moveOutputToRightbar() after the elements exist
        document.body.appendChild(rightbar);

        // Try to move output elements immediately, and retry if not ready
        moveOutputToRightbar();
        setTimeout(moveOutputToRightbar, 1000);
        setTimeout(moveOutputToRightbar, 3000);
        setTimeout(moveOutputToRightbar, 5000);
    }

    // ============================================================
    // Move output image buttons to right sidebar
    // ============================================================
    function moveOutputToRightbar() {
        const rightbar = document.getElementById('neo-rightbar');
        if (!rightbar) return;

        const tabNames = ['txt2img', 'img2img'];

        for (const tab of tabNames) {
            const buttonsId = `image_buttons_${tab}`;
            const buttonsRow = document.getElementById(buttonsId);

            // Only process if the container for this tab doesn't exist yet
            const tabContainerId = `neo-rightbar-${tab}`;
            if (document.getElementById(tabContainerId)) continue;
            if (!buttonsRow) continue;

            // Create tab container
            const tabContainer = document.createElement('div');
            tabContainer.id = tabContainerId;
            tabContainer.className = 'neo-rightbar-tab';
            tabContainer.style.display = tab === 'txt2img' ? 'flex' : 'none';

            const actionsWrap = document.createElement('div');
            actionsWrap.className = 'neo-rightbar-actions';

            // Action button definitions with icons and names
            const actionMap = [
                { id: `${tab}_open_folder`, icon: '📂', name: '打开目录' },
                { id: `save_${tab}`, icon: '💾', name: '保存' },
                { id: `save_zip_${tab}`, icon: '🗃️', name: '打包下载' },
                { id: `${tab}_send_to_img2img`, icon: '🖼️', name: '发送到图生图' },
                { id: `${tab}_send_to_inpaint`, icon: '🎨', name: '发送到重绘' },
                { id: `${tab}_send_to_extras`, icon: '📐', name: '发送到后期' },
            ];

            // Add upscale button for txt2img
            if (tab === 'txt2img') {
                actionMap.push({ id: `${tab}_upscale`, icon: '✨', name: '高清放大' });
            }

            for (const action of actionMap) {
                const originalBtn = document.getElementById(action.id);
                if (!originalBtn) continue;

                const wrapper = document.createElement('button');
                wrapper.className = 'neo-rightbar-action-btn';
                wrapper.setAttribute('data-original-id', action.id);
                wrapper.innerHTML = `<span class="btn-icon">${action.icon}</span><span class="btn-label">${action.name}</span>`;

                // Forward click events to the original button
                wrapper.addEventListener('click', function (e) {
                    e.preventDefault();
                    e.stopPropagation();
                    const orig = document.getElementById(this.getAttribute('data-original-id'));
                    if (orig) orig.click();
                });

                actionsWrap.appendChild(wrapper);
            }

            tabContainer.appendChild(actionsWrap);
            rightbar.appendChild(tabContainer);

            // Hide the original buttons row
            buttonsRow.style.display = 'none';
        }

        // Show/hide right sidebar tab sections based on active tab
        const activeTab = document.querySelector('.tab-nav .tab-nav-item.selected');
        if (activeTab) {
            const tabId = activeTab.getAttribute('id');
            if (tabId) {
                showRightbarTab(tabId.includes('txt2img') ? 'txt2img' : 'img2img');
            }
        }
    }

    function showRightbarTab(tabName) {
        const txt2imgContainer = document.getElementById('neo-rightbar-txt2img');
        const img2imgContainer = document.getElementById('neo-rightbar-img2img');
        if (txt2imgContainer) txt2imgContainer.style.display = tabName === 'txt2img' ? 'flex' : 'none';
        if (img2imgContainer) img2imgContainer.style.display = tabName === 'img2img' ? 'flex' : 'none';
    }

    // ============================================================
    // Create bottom bar for generation info & run log
    // ============================================================
    function createBottombar() {
        if (document.getElementById('neo-bottombar')) return;

        const bottombar = document.createElement('div');
        bottombar.id = 'neo-bottombar';
        document.body.appendChild(bottombar);

        // Try to populate immediately, and retry if elements not ready
        populateBottombar();
        setTimeout(populateBottombar, 1000);
        setTimeout(populateBottombar, 3000);
        setTimeout(populateBottombar, 5000);
    }

    function populateBottombar() {
        const bottombar = document.getElementById('neo-bottombar');
        if (!bottombar) return;

        // Only populate once
        if (bottombar.hasAttribute('data-populated')) return;

        // Start collapsed
        bottombar.classList.add('collapsed');

        const tabNames = ['txt2img', 'img2img'];
        let hasContent = false;

        // --- Create header row (tabs + toggle) ---
        const header = document.createElement('div');
        header.className = 'neo-bottombar-header';

        const tabsHeader = document.createElement('div');
        tabsHeader.className = 'neo-bottombar-tabs';
        header.appendChild(tabsHeader);

        // Toggle button
        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'neo-bottombar-toggle';
        toggleBtn.innerHTML = '▲';
        toggleBtn.title = '展开/折叠';
        toggleBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            toggleBottombar();
        });
        header.appendChild(toggleBtn);

        bottombar.appendChild(header);

        // --- Create body (content area) ---
        const body = document.createElement('div');
        body.className = 'neo-bottombar-body';

        // Create info container (generation info + run log combined)
        const infoContainer = document.createElement('div');
        infoContainer.id = 'neo-bottombar-info';
        infoContainer.className = 'active';
        body.appendChild(infoContainer);

        // Create version container (footer)
        const versionContainer = document.createElement('div');
        versionContainer.id = 'neo-bottombar-version';
        body.appendChild(versionContainer);

        // Create history container (generation history)
        const historyContainer = document.createElement('div');
        historyContainer.id = 'neo-bottombar-history';
        historyContainer.className = 'neo-history-grid';
        body.appendChild(historyContainer);

        bottombar.appendChild(body);

        // Move info and log elements into the info container (combined)
        for (const tab of tabNames) {
            const infoEl = document.getElementById(`html_info_${tab}`);
            const logEl = document.getElementById(`html_log_${tab}`);

            if (infoEl) {
                infoEl.setAttribute('data-tab', tab);
                infoContainer.appendChild(infoEl);
                hasContent = true;
            }
            if (logEl) {
                logEl.setAttribute('data-tab', tab);
                infoContainer.appendChild(logEl);
                hasContent = true;
            }
        }

        // Move footer element into the version container
        const footerEl = document.getElementById('footer');
        if (footerEl) {
            versionContainer.appendChild(footerEl);
            hasContent = true;
        }

        if (!hasContent) {
            bottombar.innerHTML = '';
            return;
        }

        // Create tab buttons
        const tabInfo = document.createElement('button');
        tabInfo.className = 'neo-bottombar-tab-btn active';
        tabInfo.textContent = '生成信息';
        tabInfo.addEventListener('click', function () {
            document.querySelectorAll('.neo-bottombar-tab-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            document.querySelectorAll('.neo-bottombar-body > div').forEach(d => d.classList.remove('active'));
            const info = document.getElementById('neo-bottombar-info');
            if (info) info.classList.add('active');
        });
        tabsHeader.appendChild(tabInfo);

        const tabVersion = document.createElement('button');
        tabVersion.className = 'neo-bottombar-tab-btn';
        tabVersion.textContent = '版本信息';
        tabVersion.addEventListener('click', function () {
            document.querySelectorAll('.neo-bottombar-tab-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            document.querySelectorAll('.neo-bottombar-body > div').forEach(d => d.classList.remove('active'));
            const version = document.getElementById('neo-bottombar-version');
            if (version) version.classList.add('active');
        });
        tabsHeader.appendChild(tabVersion);

        const tabHistory = document.createElement('button');
        tabHistory.className = 'neo-bottombar-tab-btn';
        tabHistory.textContent = '生成历史';
        tabHistory.addEventListener('click', function () {
            document.querySelectorAll('.neo-bottombar-tab-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            document.querySelectorAll('.neo-bottombar-body > div').forEach(d => d.classList.remove('active'));
            const history = document.getElementById('neo-bottombar-history');
            if (history) {
                history.classList.add('active');
                loadGenerationHistory();
            }
        });
        tabsHeader.appendChild(tabHistory);

        // Show only the active tab's content
        showBottombarTab('txt2img');

        // Click on header toggles expand/collapse (but not on tab buttons)
        header.addEventListener('click', function (e) {
            // Don't toggle if clicking a tab button or the toggle button itself
            if (e.target.closest('.neo-bottombar-tab-btn') || e.target.closest('.neo-bottombar-toggle')) return;
            toggleBottombar();
        });

        // Restore saved state from localStorage
        try {
            const saved = localStorage.getItem('neo_bottombar_collapsed');
            if (saved === '0') {
                // Was expanded before — expand now
                toggleBottombar();
            }
        } catch (e) {}

        bottombar.setAttribute('data-populated', 'true');
    }

    // ============================================================
    // Generation History Functions
    // ============================================================
    let historyLoaded = false;

    function loadGenerationHistory() {
        if (historyLoaded) return;
        historyLoaded = true;

        const container = document.getElementById('neo-bottombar-history');
        if (!container) return;

        // Show loading indicator
        container.innerHTML = '<div class="neo-history-loading">加载中...</div>';

        fetch('/neo-history/images?limit=100&tab=all')
            .then(function (resp) {
                if (!resp.ok) throw new Error('HTTP ' + resp.status);
                return resp.json();
            })
            .then(function (data) {
                if (data.images && data.images.length > 0) {
                    renderHistoryThumbnails(data.images);
                } else {
                    container.innerHTML = '<div class="neo-history-empty">暂无生成历史 - 生成图像后刷新此页面查看</div>';
                }
            })
            .catch(function (err) {
                console.error('[NEO] Failed to load generation history:', err);
                container.innerHTML = '<div class="neo-history-empty">加载失败，请确认WebUI已重启且API已注册</div>';
                historyLoaded = false; // Allow retry
            });
    }

    function renderHistoryThumbnails(images) {
        const container = document.getElementById('neo-bottombar-history');
        if (!container) return;
        container.innerHTML = '';
        container.classList.remove('neo-history-loading');

        // Create filter bar
        const filterBar = document.createElement('div');
        filterBar.className = 'neo-history-filter';

        const filterLabel = document.createElement('span');
        filterLabel.textContent = '筛选: ';
        filterBar.appendChild(filterLabel);

        const btnAll = document.createElement('button');
        btnAll.className = 'neo-history-filter-btn active';
        btnAll.textContent = '全部';
        btnAll.addEventListener('click', function () {
            document.querySelectorAll('.neo-history-filter-btn').forEach(function (b) { b.classList.remove('active'); });
            this.classList.add('active');
            filterHistory('all');
        });
        filterBar.appendChild(btnAll);

        const btnTxt2img = document.createElement('button');
        btnTxt2img.className = 'neo-history-filter-btn';
        btnTxt2img.textContent = '文生图';
        btnTxt2img.addEventListener('click', function () {
            document.querySelectorAll('.neo-history-filter-btn').forEach(function (b) { b.classList.remove('active'); });
            this.classList.add('active');
            filterHistory('txt2img');
        });
        filterBar.appendChild(btnTxt2img);

        const btnImg2img = document.createElement('button');
        btnImg2img.className = 'neo-history-filter-btn';
        btnImg2img.textContent = '图生图';
        btnImg2img.addEventListener('click', function () {
            document.querySelectorAll('.neo-history-filter-btn').forEach(function (b) { b.classList.remove('active'); });
            this.classList.add('active');
            filterHistory('img2img');
        });
        filterBar.appendChild(btnImg2img);

        container.appendChild(filterBar);

        // Create grid
        const grid = document.createElement('div');
        grid.className = 'neo-history-grid-inner';
        grid.id = 'neo-history-grid-inner';
        container.appendChild(grid);

        // Store all images data for filtering
        container._allImages = images;

        // Render all images
        for (var i = 0; i < images.length; i++) {
            var img = images[i];
            var thumb = document.createElement('div');
            thumb.className = 'neo-history-thumb';
            thumb.dataset.type = img.type;

            var imgEl = document.createElement('img');
            imgEl.src = '/neo-history/image/' + encodeURIComponent(img.path);
            imgEl.alt = img.filename;
            imgEl.loading = 'lazy';

            var label = document.createElement('div');
            label.className = 'neo-history-thumb-label';
            label.textContent = img.type === 'txt2img' ? '文' : '图';

            thumb.appendChild(imgEl);
            thumb.appendChild(label);

            thumb.addEventListener('click', function () {
                var src = this.querySelector('img').src;
                var name = this.querySelector('img').alt;
                openHistoryModal(src, name);
            });

            grid.appendChild(thumb);
        }
    }

    function filterHistory(type) {
        var grid = document.getElementById('neo-history-grid-inner');
        var container = document.getElementById('neo-bottombar-history');
        if (!grid || !container) return;

        var images = container._allImages || [];
        grid.innerHTML = '';

        for (var i = 0; i < images.length; i++) {
            var img = images[i];
            if (type !== 'all' && img.type !== type) continue;

            var thumb = document.createElement('div');
            thumb.className = 'neo-history-thumb';
            thumb.dataset.type = img.type;

            var imgEl = document.createElement('img');
            imgEl.src = '/neo-history/image/' + encodeURIComponent(img.path);
            imgEl.alt = img.filename;
            imgEl.loading = 'lazy';

            var label = document.createElement('div');
            label.className = 'neo-history-thumb-label';
            label.textContent = img.type === 'txt2img' ? '文' : '图';

            thumb.appendChild(imgEl);
            thumb.appendChild(label);

            thumb.addEventListener('click', function () {
                var src = this.querySelector('img').src;
                var name = this.querySelector('img').alt;
                openHistoryModal(src, name);
            });

            grid.appendChild(thumb);
        }
    }

    function openHistoryModal(imgSrc, filename) {
        // Remove existing modal
        var existing = document.getElementById('neo-history-modal');
        if (existing) existing.remove();

        var overlay = document.createElement('div');
        overlay.id = 'neo-history-modal';
        overlay.className = 'neo-history-modal-overlay';

        var modal = document.createElement('div');
        modal.className = 'neo-history-modal';

        var closeBtn = document.createElement('button');
        closeBtn.className = 'neo-history-modal-close';
        closeBtn.innerHTML = '✕';
        closeBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            overlay.remove();
        });

        var imgEl = document.createElement('img');
        imgEl.className = 'neo-history-modal-img';
        imgEl.src = imgSrc;
        imgEl.alt = filename;

        modal.appendChild(closeBtn);
        modal.appendChild(imgEl);
        overlay.appendChild(modal);
        document.body.appendChild(overlay);

        // Close on overlay click
        overlay.addEventListener('click', function (e) {
            if (e.target === overlay) overlay.remove();
        });
    }

    function toggleBottombar() {
        const bottombar = document.getElementById('neo-bottombar');
        if (!bottombar) return;
        const isCollapsed = bottombar.classList.toggle('collapsed');

        // Save state to localStorage
        try {
            localStorage.setItem('neo_bottombar_collapsed', isCollapsed ? '1' : '0');
        } catch (e) {}

        // Update main content margin to make room
        const container = document.querySelector('.gradio-container');
        if (container) {
            container.style.marginBottom = isCollapsed ? '32px' : '500px';
        }
    }

    // Show/hide bottom bar content for the active tab (txt2img vs img2img)
    function showBottombarTab(tabName) {
        const infoContainer = document.getElementById('neo-bottombar-info');
        const versionContainer = document.getElementById('neo-bottombar-version');
        if (!infoContainer && !versionContainer) return;

        // Hide all tab-specific elements
        document.querySelectorAll('#neo-bottombar [data-tab]').forEach(el => {
            el.style.display = 'none';
        });

        // Show the active tab's elements
        document.querySelectorAll(`#neo-bottombar [data-tab="${tabName}"]`).forEach(el => {
            el.style.display = '';
        });
    }

    // ============================================================
    // Main initialization
    // ============================================================
    function init() {
        if (initialized) return;
        if (!document.querySelector('#tabs')) {
            setTimeout(init, 300);
            return;
        }

        console.log('[NEO] Initializing...');

        // Fix corrupted visibility state: if no topbar item is explicitly saved as visible,
        // clear the state so all items show by default
        (function fixCorruptedState() {
            const saved = loadVisibleItems();
            if (!saved) return;
            let hasAnyTopbarVisible = false;
            for (const item of TOPBAR_ITEMS) {
                if (item.type === 'separator') continue;
                if (saved[item.id] === true) { hasAnyTopbarVisible = true; break; }
            }
            if (!hasAnyTopbarVisible && Object.keys(saved).length > 0) {
                console.log('[NEO] Migration: cleared visibility state so all items show by default');
                localStorage.removeItem(STORAGE_KEY_VISIBLE_ITEMS);
            }
        })();

        // Apply saved item order from localStorage
        applySavedOrder();

        createTopbar();
        createSidebar();
        createRightbar();
        createBottombar();
        hideTabs();
        hideAllPanels();
        applyTopbarPanelState();

        setTimeout(retryHidePanels, 1500);
        setTimeout(retryHidePanels, 3000);
        setTimeout(retryHidePanels, 6000);
        setTimeout(retryHidePanels, 10000);

        setTimeout(applyTopbarPanelState, 1500);
        setTimeout(applyTopbarPanelState, 3000);
        setTimeout(applyTopbarPanelState, 6000);
        setTimeout(applyTopbarPanelState, 10000);

        setInterval(function () {
            const savedVisibleState = loadVisibleItems();
            for (const item of SIDEBAR_ITEMS) {
                if (!isItemInstalled(item)) continue;
                if (item.type === 'accordion') {
                    if (accordionContainers[item.id] && accordionContainers[item.id].length > 0) continue;
                    const wasVisible = savedVisibleState && savedVisibleState[item.id] === true;
                    if (wasVisible) continue;
                    const containers = findAccordionContainers(item.match);
                    if (containers.length > 0) {
                        accordionContainers[item.id] = containers;
                        for (const c of containers) {
                            c.style.display = 'none';
                            c.classList.add('neo-sidebar-hidden-panel');
                        }
                        console.log(`[NEO] ✓ (periodic) Hidden accordion "${item.label}"`);
                    }
                }
            }
        }, 5000);

        setupTabObserver();
        initialized = true;
        console.log('[NEO] Initialization complete');
    }

    // --- Start ---
    if (document.readyState === 'complete') {
        setTimeout(init, 200);
    } else {
        window.addEventListener('load', function () {
            setTimeout(init, 200);
        });
    }

})();