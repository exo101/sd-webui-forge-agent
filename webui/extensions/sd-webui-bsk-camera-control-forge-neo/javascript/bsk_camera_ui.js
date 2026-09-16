// SPDX-FileCopyrightText: 2026 灰暗x
// SPDX-FileCopyrightText: 2026 Forge Neo port contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Forge Neo browser UI port of ComfyUI-bsk_UI's CameraControlNode.
// Original project: https://github.com/ikusag-png/ComfyUI_bsk_UI

(function bootstrap(attempt) {
    "use strict";

    const core = window.BskForgeCameraCore;
    if (!core) {
        if (attempt < 100) {
            window.setTimeout(() => bootstrap(attempt + 1), 50);
        } else {
            console.error("[BSK Camera] bsk_camera_core.js was not loaded.");
        }
        return;
    }

    const EXTENSION_ID = "bsk-camera-forge-neo";
    const STORAGE_PREFIX = "bsk_camera_forge_neo.v1";
    const CANVAS_WIDTH = 560;
    const CANVAS_HEIGHT = 500;
    const TARGETS = [
        {name: "txt2img", label: "文生图"},
        {name: "img2img", label: "图生图"},
    ];
    const SUBMIT_HOOKS = [
        {
            functionName: "submit",
            targetName: "txt2img",
            promptIndex: 1,
            negativePromptIndex: 2,
        },
        {
            functionName: "submit_img2img",
            targetName: "img2img",
            promptIndex: 2,
            negativePromptIndex: 3,
        },
    ];
    const SUBMIT_WRAPPER_MARKER = "__bskCameraForgeNeoSubmitWrapper";
    const EXTRA_LABELS = {
        lens: "镜头 / 焦距",
        dof: "景深",
        movement: "运镜",
        composition: "构图",
        style: "风格",
    };
    const generationBridge = {
        targets: new Map(),
        wrappers: new Map(),
        retryTimer: null,
        retryCount: 0,
    };

    function getAppRoot() {
        return typeof window.gradioApp === "function" ? window.gradioApp() : document;
    }

    function storageKey(target, part) {
        return `${STORAGE_PREFIX}.${target}.${part}`;
    }

    function readStorage(key, fallback) {
        try {
            const raw = window.localStorage.getItem(key);
            return raw === null ? fallback : JSON.parse(raw);
        } catch (_error) {
            return fallback;
        }
    }

    function writeStorage(key, value) {
        try {
            window.localStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch (error) {
            console.warn("[BSK Camera] Could not save browser settings:", error);
            return false;
        }
    }

    function createElement(tag, className, text) {
        const element = document.createElement(tag);
        if (className) {
            element.className = className;
        }
        if (text !== undefined) {
            element.textContent = text;
        }
        return element;
    }

    function createButton(text, className = "") {
        const button = createElement(
            "button",
            `bsk-camera-button ${className}`.trim(),
            text,
        );
        button.type = "button";
        return button;
    }

    function createDetails(title, open = false) {
        const details = createElement("details", "bsk-camera-subsection");
        details.open = open;
        const summary = createElement("summary", "", title);
        details.appendChild(summary);
        return details;
    }

    function showToast(message, kind = "info") {
        let host = document.getElementById(`${EXTENSION_ID}-toasts`);
        if (!host) {
            host = createElement("div", "bsk-camera-toasts");
            host.id = `${EXTENSION_ID}-toasts`;
            document.body.appendChild(host);
        }

        const toast = createElement(
            "div",
            `bsk-camera-toast bsk-camera-toast-${kind}`,
            message,
        );
        host.appendChild(toast);
        window.setTimeout(() => toast.remove(), 2200);
    }

    async function copyText(text) {
        if (navigator.clipboard?.writeText) {
            try {
                await navigator.clipboard.writeText(text);
                return true;
            } catch (_error) {
                // Use the older browser fallback below.
            }
        }

        const helper = document.createElement("textarea");
        helper.value = text;
        helper.setAttribute("readonly", "");
        helper.style.position = "fixed";
        helper.style.opacity = "0";
        document.body.appendChild(helper);
        helper.select();

        let copied = false;
        try {
            copied = document.execCommand("copy");
        } catch (_error) {
            copied = false;
        }
        helper.remove();
        return copied;
    }

    async function readClipboardOrPrompt() {
        if (navigator.clipboard?.readText) {
            try {
                const text = await navigator.clipboard.readText();
                if (text) {
                    return text;
                }
            } catch (_error) {
                // Browser permission denied; fall back to a paste dialog.
            }
        }

        return window.prompt("请粘贴机位配置 JSON：", "") || "";
    }

    function findPromptContainer(targetName) {
        return getAppRoot().querySelector(`#${targetName}_prompt`);
    }

    function joinPrompt(base, cameraPrompt) {
        const cleanBase = String(base || "").trim();
        const cleanCamera = String(cameraPrompt || "").trim();

        if (!cleanBase) {
            return cleanCamera;
        }
        if (!cleanCamera) {
            return cleanBase;
        }
        return `${cleanBase.replace(/[\s,]+$/u, "")}, ${cleanCamera}`;
    }

    function mergeSubmissionArgs(hook, args) {
        const registration = generationBridge.targets.get(hook.targetName);
        if (!registration || !registration.isEnabled()) {
            return args;
        }

        const prompt = args[hook.promptIndex];
        const negativePrompt = args[hook.negativePromptIndex];
        const cameraPrompt = String(registration.getPrompt() || "").trim();
        if (
            typeof prompt !== "string" ||
            typeof negativePrompt !== "string" ||
            !cameraPrompt
        ) {
            return args;
        }

        const merged = joinPrompt(prompt, cameraPrompt);
        if (merged === prompt) {
            return args;
        }

        const mergedArgs = Array.from(args);
        mergedArgs[hook.promptIndex] = merged;
        return mergedArgs;
    }

    function wrapSubmitFunction(hook) {
        if (generationBridge.wrappers.has(hook.functionName)) {
            return true;
        }

        const original = window[hook.functionName];
        if (typeof original !== "function") {
            return false;
        }

        if (original[SUBMIT_WRAPPER_MARKER]?.extensionId === EXTENSION_ID) {
            generationBridge.wrappers.set(hook.functionName, original);
            return true;
        }

        function wrappedSubmit(...args) {
            const submissionArgs =
                mergeSubmissionArgs(hook, args) || args;
            return Reflect.apply(original, this, submissionArgs);
        }

        Object.defineProperty(wrappedSubmit, SUBMIT_WRAPPER_MARKER, {
            value: {
                extensionId: EXTENSION_ID,
                targetName: hook.targetName,
                original,
            },
        });

        window[hook.functionName] = wrappedSubmit;
        if (window[hook.functionName] !== wrappedSubmit) {
            return false;
        }

        generationBridge.wrappers.set(hook.functionName, wrappedSubmit);
        return true;
    }

    function ensureGenerationHooks() {
        const ready = SUBMIT_HOOKS.every(wrapSubmitFunction);
        if (ready || generationBridge.retryTimer !== null) {
            return;
        }

        if (generationBridge.retryCount >= 100) {
            console.error(
                "[BSK Camera] Forge submit functions were not available; " +
                    "camera prompts cannot be merged into generation requests.",
            );
            return;
        }

        generationBridge.retryCount += 1;
        generationBridge.retryTimer = window.setTimeout(() => {
            generationBridge.retryTimer = null;
            ensureGenerationHooks();
        }, 50);
    }

    function registerGenerationTarget(targetName, registration) {
        generationBridge.targets.set(targetName, registration);
        ensureGenerationHooks();
    }

    function numericValue(value, fallback = 0) {
        const parsed = Number.parseFloat(value);
        return Number.isFinite(parsed) ? parsed : fallback;
    }

    function normalizedWheelDelta(event) {
        let delta = event.deltaY;
        if (event.deltaMode === 1) {
            delta *= 16;
        } else if (event.deltaMode === 2) {
            delta *= 100;
        }
        return delta;
    }

    function mountTarget(target) {
        const promptContainer = findPromptContainer(target.name);
        if (!promptContainer) {
            return;
        }

        const panelId = `${EXTENSION_ID}-${target.name}`;
        if (getAppRoot().querySelector(`#${panelId}`)) {
            return;
        }

        let config = core.loadConfig(
            readStorage(storageKey(target.name, "config"), null),
        );
        let state = core.normalizeState(
            readStorage(storageKey(target.name, "state"), null),
        );
        const storedUiState = readStorage(
            storageKey(target.name, "ui"),
            {},
        );
        let uiState = {
            open: storedUiState.open === true,
            mergeOnGenerate: storedUiState.mergeOnGenerate !== false,
        };
        let currentOutput = core.computeFromState(state, config);
        let renderFrame = null;

        const panel = createElement("details", "bsk-camera-extension");
        panel.id = panelId;
        panel.open = uiState.open === true;

        const summary = createElement("summary", "bsk-camera-main-summary");
        const summaryTitle = createElement(
            "span",
            "bsk-camera-summary-title",
            `机位控制 · ${target.label}`,
        );
        const summaryOutput = createElement(
            "span",
            "bsk-camera-summary-output",
            "准备就绪",
        );
        summary.append(summaryTitle, summaryOutput);
        panel.appendChild(summary);

        const body = createElement("div", "bsk-camera-body");
        panel.appendChild(body);

        const toolbar = createElement("div", "bsk-camera-toolbar");
        const copyOutputButton = createButton("复制机位词");
        const resetPositionButton = createButton("机位归零");

        const mergeOnGenerateLabel = createElement(
            "label",
            "bsk-camera-checkbox-label master",
        );
        const mergeOnGenerateCheckbox = document.createElement("input");
        mergeOnGenerateCheckbox.type = "checkbox";
        mergeOnGenerateCheckbox.checked = uiState.mergeOnGenerate;
        mergeOnGenerateLabel.title =
            "只在生成任务的提交参数中与正向提示词合并，不改写正向提示词框";
        mergeOnGenerateLabel.append(
            mergeOnGenerateCheckbox,
            createElement("span", "", "生成时融合机位提示词"),
        );

        toolbar.append(
            mergeOnGenerateLabel,
            copyOutputButton,
            resetPositionButton,
        );
        body.appendChild(toolbar);

        const workspace = createElement("div", "bsk-camera-workspace");
        const visualColumn = createElement("div", "bsk-camera-visual-column");
        const settingsColumn = createElement("div", "bsk-camera-settings-column");
        workspace.append(visualColumn, settingsColumn);
        body.appendChild(workspace);

        const legalNotice = createElement("div", "bsk-camera-legal");
        const sourceLink = document.createElement("a");
        sourceLink.href = "https://github.com/ikusag-png/ComfyUI_bsk_UI";
        sourceLink.target = "_blank";
        sourceLink.rel = "noreferrer";
        sourceLink.textContent = "ComfyUI-bsk_UI";
        legalNotice.append(
            document.createTextNode("机位控件移植自 "),
            sourceLink,
            document.createTextNode(" · © 2026 B站 @灰暗x · AGPL-3.0-or-later · 不提供任何担保"),
        );
        body.appendChild(legalNotice);

        const canvasShell = createElement("div", "bsk-camera-canvas-shell");
        const canvas = document.createElement("canvas");
        const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = CANVAS_WIDTH * pixelRatio;
        canvas.height = CANVAS_HEIGHT * pixelRatio;
        canvas.className = "bsk-camera-canvas";
        canvas.tabIndex = 0;
        canvas.setAttribute(
            "aria-label",
            "机位控制画布：拖动调整方位与高度，滚轮调整距离，Shift 加滚轮调整倾斜",
        );

        const canvasHint = createElement(
            "div",
            "bsk-camera-canvas-hint",
            "拖拽画布 · 抓住世界旋转",
        );
        const canvasReadout = createElement("div", "bsk-camera-canvas-readout");
        const directionReadout = createElement("div", "direction", "FRONT · 0°");
        const rollReadout = createElement("div", "roll", "ROLL · 0°");
        canvasReadout.append(directionReadout, rollReadout);
        canvasShell.append(canvas, canvasHint, canvasReadout);
        visualColumn.appendChild(canvasShell);

        const axisControls = createElement("div", "bsk-camera-axis-controls");
        const axisBindings = {};

        function updateStateValue(key, value) {
            state[key] = core.clamp(numericValue(value, 0), -1, 1);
            scheduleRender();
        }

        function createAxisControl(label, key) {
            const row = createElement("div", "bsk-camera-axis-row");
            const name = createElement("label", "", label);
            const slider = document.createElement("input");
            slider.type = "range";
            slider.min = "-1";
            slider.max = "1";
            slider.step = "0.01";
            slider.value = state[key];

            const number = document.createElement("input");
            number.type = "number";
            number.min = "-1";
            number.max = "1";
            number.step = "0.01";
            number.value = state[key].toFixed(2);

            slider.addEventListener("input", () => updateStateValue(key, slider.value));
            number.addEventListener("change", () => updateStateValue(key, number.value));
            number.addEventListener("keydown", (event) => {
                if (event.key === "Enter") {
                    updateStateValue(key, number.value);
                    number.blur();
                }
            });

            row.append(name, slider, number);
            axisBindings[key] = {slider, number};
            return row;
        }

        axisControls.append(
            createAxisControl("左右 (X)", "pos_x"),
            createAxisControl("上下 (Y)", "pos_y"),
            createAxisControl("远近 (Z)", "pos_z"),
            createAxisControl("翻滚 (R)", "roll"),
        );
        visualColumn.appendChild(axisControls);

        const visualActions = createElement("div", "bsk-camera-visual-actions");
        const dragModeButton = createButton(
            state.drag_mode === "absolute" ? "绝对拖拽" : "相对拖拽",
        );
        const visualResetButton = createButton("归位 X/Y/Z/R");
        visualActions.append(dragModeButton, visualResetButton);
        visualColumn.appendChild(visualActions);

        const outputLabel = createElement(
            "label",
            "bsk-camera-output-label",
            "独立机位提示词",
        );
        const output = document.createElement("textarea");
        output.id = `${EXTENSION_ID}-${target.name}-prompt`;
        output.className = "bsk-camera-output";
        output.readOnly = true;
        output.rows = 3;
        output.value = currentOutput;
        outputLabel.htmlFor = output.id;
        output.title =
            "此框与正向提示词分开存放；生成时只在提交参数中合并";
        output.addEventListener("focus", () => output.select());
        visualColumn.append(outputLabel, output);

        const mergeHelp = createElement(
            "div",
            "bsk-camera-merge-help",
            "不会改写正向提示词框。启用“生成时融合”后，机位词只会加入本次生成任务的提交参数。",
        );
        visualColumn.appendChild(mergeHelp);

        const interactionHelp = createElement(
            "div",
            "bsk-camera-help",
            "拖拽：方位/高度　滚轮：距离　Shift+滚轮：倾斜　方向键：微调　[ ]：距离　, .：倾斜",
        );
        visualColumn.appendChild(interactionHelp);

        function persistConfig() {
            writeStorage(storageKey(target.name, "config"), config);
        }

        function persistState() {
            writeStorage(storageKey(target.name, "state"), state);
        }

        function persistUiState() {
            writeStorage(storageKey(target.name, "ui"), uiState);
        }

        registerGenerationTarget(target.name, {
            getPrompt: () => core.computeFromState(state, config),
            isEnabled: () =>
                panel.isConnected && uiState.mergeOnGenerate === true,
        });

        function rebuildPanel() {
            generationBridge.targets.delete(target.name);
            panel.remove();
            window.setTimeout(() => mountTarget(target), 0);
        }

        function onConfigChanged() {
            persistConfig();
            scheduleRender();
        }

        function createCheckbox(checked, onChange, text, title = "") {
            const label = createElement("label", "bsk-camera-checkbox-label");
            if (title) {
                label.title = title;
            }
            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.checked = checked;
            checkbox.addEventListener("change", () => onChange(checkbox.checked));
            label.append(checkbox, createElement("span", "", text));
            return {label, checkbox};
        }

        function createNumericControl(
            labelText,
            value,
            minimum,
            maximum,
            step,
            onChange,
        ) {
            const row = createElement("div", "bsk-camera-setting-row numeric");
            const label = createElement("label", "", labelText);
            const slider = document.createElement("input");
            slider.type = "range";
            slider.min = String(minimum);
            slider.max = String(maximum);
            slider.step = String(step);
            slider.value = String(value);

            const number = document.createElement("input");
            number.type = "number";
            number.min = String(minimum);
            number.max = String(maximum);
            number.step = String(step);
            number.value =
                step < 0.001
                    ? Number(value).toFixed(4)
                    : Number(value).toFixed(2);

            function commit(raw) {
                const parsed = core.clamp(
                    numericValue(raw, value),
                    minimum,
                    maximum,
                );
                slider.value = String(parsed);
                number.value =
                    step < 0.001 ? parsed.toFixed(4) : parsed.toFixed(2);
                onChange(parsed);
            }

            slider.addEventListener("input", () => commit(slider.value));
            number.addEventListener("change", () => commit(number.value));
            row.append(label, slider, number);
            return row;
        }

        function createTextSetting(
            labelText,
            item,
            valueKey = "tag",
            title = "",
        ) {
            const row = createElement("div", "bsk-camera-setting-row text");
            const enabled = createCheckbox(
                item.enabled !== false,
                (checked) => {
                    item.enabled = checked;
                    onConfigChanged();
                },
                "",
            );
            enabled.label.classList.add("compact");

            const label = createElement("label", "", labelText);
            if (title) {
                label.title = title;
            }
            const input = document.createElement("input");
            input.type = "text";
            input.value = item[valueKey] || "";
            input.title = title;
            input.addEventListener("input", () => {
                item[valueKey] = input.value;
                onConfigChanged();
            });
            row.append(enabled.label, label, input);
            return row;
        }

        function createTagFieldset(title, master, entries) {
            const fieldset = document.createElement("fieldset");
            fieldset.className = "bsk-camera-fieldset";
            const legend = document.createElement("legend");
            legend.textContent = title;
            fieldset.appendChild(legend);

            if (master) {
                const masterToggle = createCheckbox(
                    master.enabled !== false,
                    (checked) => {
                        master.enabled = checked;
                        onConfigChanged();
                    },
                    "总开关",
                );
                masterToggle.label.classList.add("master");
                fieldset.appendChild(masterToggle.label);
            }

            for (const entry of entries) {
                fieldset.appendChild(
                    createTextSetting(
                        entry.label,
                        entry.item,
                        entry.valueKey || "tag",
                        entry.title || "",
                    ),
                );
            }
            return fieldset;
        }

        const weightsDetails = createDetails("权重与手势设置", false);
        const weightsBody = createElement("div", "bsk-camera-section-body");
        weightsDetails.appendChild(weightsBody);

        const noWeightToggle = createCheckbox(
            config.no_weight === true,
            (checked) => {
                config.no_weight = checked;
                onConfigChanged();
            },
            "无权重模式（输出纯提示词）",
        );
        weightsBody.appendChild(noWeightToggle.label);
        weightsBody.append(
            createNumericControl(
                "无权重·次方向阈值",
                config.no_weight_threshold,
                0,
                1,
                0.01,
                (value) => {
                    config.no_weight_threshold = value;
                    onConfigChanged();
                },
            ),
            createNumericControl(
                "方位整体权重",
                config.azimuth.weight,
                0.1,
                10,
                0.1,
                (value) => {
                    config.azimuth.weight = value;
                    onConfigChanged();
                },
            ),
            createNumericControl(
                "权重最大值",
                config.weight_max,
                1,
                10,
                0.1,
                (value) => {
                    config.weight_max = value;
                    onConfigChanged();
                },
            ),
            createNumericControl(
                "额外权重总控",
                config.extra_master,
                0.1,
                10,
                0.1,
                (value) => {
                    config.extra_master = value;
                    onConfigChanged();
                },
            ),
            createNumericControl(
                "高度额外权重",
                config.elevation.extra,
                -10,
                10,
                0.1,
                (value) => {
                    config.elevation.extra = value;
                    onConfigChanged();
                },
            ),
            createNumericControl(
                "平视权重（正平视时）",
                config.elevation.eye_peak,
                0.1,
                10,
                0.1,
                (value) => {
                    config.elevation.eye_peak = value;
                    onConfigChanged();
                },
            ),
            createNumericControl(
                "距离额外权重",
                config.distance.extra,
                -10,
                10,
                0.1,
                (value) => {
                    config.distance.extra = value;
                    onConfigChanged();
                },
            ),
            createNumericControl(
                "倾斜额外权重",
                config.tilt.extra,
                -10,
                10,
                0.1,
                (value) => {
                    config.tilt.extra = value;
                    onConfigChanged();
                },
            ),
            createNumericControl(
                "方位死区",
                config.azimuth.deadzone_ratio,
                0,
                0.9,
                0.01,
                (value) => {
                    config.azimuth.deadzone_ratio = value;
                    onConfigChanged();
                },
            ),
            createNumericControl(
                "倾斜死区",
                config.tilt.deadzone,
                0,
                0.9,
                0.01,
                (value) => {
                    config.tilt.deadzone = value;
                    onConfigChanged();
                },
            ),
            createNumericControl(
                "滚轮距离步进",
                config.wheel_step,
                0.0002,
                0.005,
                0.0001,
                (value) => {
                    config.wheel_step = value;
                    onConfigChanged();
                },
            ),
        );

        const promptDetails = createDetails("自定义提示词（可直接改词）", true);
        const promptBody = createElement("div", "bsk-camera-section-body");
        promptDetails.appendChild(promptBody);

        promptBody.append(
            createTagFieldset("方位", config.azimuth, [
                {
                    label: "前",
                    item: config.azimuth.directions.front,
                    title: "相机位于角色正面时输出",
                },
                {
                    label: "后",
                    item: config.azimuth.directions.back,
                    title: "相机位于角色背面时输出",
                },
                {
                    label: "左",
                    item: config.azimuth.directions.left,
                    title: "当前原版默认输出 from right；可在此改为任意词",
                },
                {
                    label: "右",
                    item: config.azimuth.directions.right,
                    title: "当前原版默认输出 from left；可在此改为任意词",
                },
            ]),
            createTagFieldset("高度", config.elevation, [
                {label: "鸟瞰", item: config.elevation.categories.bird},
                {label: "高机位", item: config.elevation.categories.high},
                {label: "平视", item: config.elevation.categories.eye},
                {label: "低机位", item: config.elevation.categories.low},
                {label: "正下方", item: config.elevation.categories.worm},
            ]),
            createTagFieldset("距离", config.distance, [
                {label: "大特写", item: config.distance.categories.ecu},
                {label: "近景", item: config.distance.categories.cu},
                {label: "中景", item: config.distance.categories.medium},
                {label: "全身", item: config.distance.categories.full},
                {label: "远景", item: config.distance.categories.wide},
            ]),
            createTagFieldset("倾斜", null, [
                {
                    label: "倾斜词",
                    item: config.tilt,
                    valueKey: "dutch_tag",
                },
            ]),
        );

        const extrasFieldset = document.createElement("fieldset");
        extrasFieldset.className = "bsk-camera-fieldset";
        const extrasLegend = document.createElement("legend");
        extrasLegend.textContent = "额外相机提示词";
        extrasFieldset.appendChild(extrasLegend);

        for (const key of ["lens", "dof", "movement", "composition", "style"]) {
            const item = config.extras[key];
            const row = createElement("div", "bsk-camera-setting-row extra");
            const enabled = createCheckbox(
                item.enabled === true,
                (checked) => {
                    item.enabled = checked;
                    onConfigChanged();
                },
                "",
            );
            enabled.label.classList.add("compact");
            const label = createElement("label", "", EXTRA_LABELS[key]);
            const input = document.createElement("input");
            input.type = "text";
            input.value = item.value || "";
            input.addEventListener("input", () => {
                item.value = input.value;
                onConfigChanged();
            });
            row.append(enabled.label, label, input);

            if (key === "dof") {
                const weight = document.createElement("input");
                weight.type = "number";
                weight.min = "0.1";
                weight.max = "10";
                weight.step = "0.1";
                weight.value = String(item.weight ?? 1.3);
                weight.title = "景深提示词权重";
                weight.className = "bsk-camera-extra-weight";
                weight.addEventListener("change", () => {
                    item.weight = core.clamp(
                        numericValue(weight.value, 1.3),
                        0.1,
                        10,
                    );
                    weight.value = String(item.weight);
                    onConfigChanged();
                });
                row.appendChild(weight);
            }
            extrasFieldset.appendChild(row);
        }
        promptBody.appendChild(extrasFieldset);

        const presetDetails = createDetails("配置 / 预设 / 导入导出", false);
        const presetBody = createElement("div", "bsk-camera-section-body");
        presetDetails.appendChild(presetBody);

        const presetRow = createElement("div", "bsk-camera-preset-row");
        const presetSelect = document.createElement("select");
        const savePresetButton = createButton("保存预设");
        const deletePresetButton = createButton("删除预设");
        presetRow.append(presetSelect, savePresetButton, deletePresetButton);
        presetBody.appendChild(presetRow);

        const configActions = createElement("div", "bsk-camera-config-actions");
        const exportButton = createButton("复制配置 JSON");
        const importButton = createButton("导入配置 JSON");
        const resetConfigButton = createButton("恢复原版默认值", "danger");
        configActions.append(exportButton, importButton, resetConfigButton);
        presetBody.appendChild(configActions);
        presetBody.appendChild(
            createElement(
                "p",
                "bsk-camera-help",
                "配置与预设保存在当前浏览器中；复制 JSON 可在其它浏览器或另一张 Forge 页面导入。",
            ),
        );

        settingsColumn.append(weightsDetails, promptDetails, presetDetails);

        function getPresets() {
            const presets = readStorage(
                storageKey(target.name, "presets"),
                {},
            );
            return presets && typeof presets === "object" ? presets : {};
        }

        function refreshPresetSelect(selected = "") {
            const presets = getPresets();
            presetSelect.innerHTML = "";
            const placeholder = document.createElement("option");
            placeholder.value = "";
            placeholder.textContent = "选择已保存预设";
            presetSelect.appendChild(placeholder);

            for (const name of Object.keys(presets).sort((a, b) =>
                a.localeCompare(b, "zh-CN"),
            )) {
                const option = document.createElement("option");
                option.value = name;
                option.textContent = name;
                presetSelect.appendChild(option);
            }
            presetSelect.value = selected;
        }

        refreshPresetSelect();

        function serializeConfiguration() {
            return {
                schema: "bsk-camera-forge-neo",
                version: 1,
                source_version: core.VERSION,
                target: target.name,
                state: core.normalizeState(state),
                config: core.loadConfig(config),
            };
        }

        function importConfiguration(raw) {
            const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
            if (!parsed || typeof parsed !== "object") {
                throw new Error("JSON 顶层必须是对象");
            }

            const importedConfig = parsed.config || parsed;
            config = core.loadConfig(importedConfig);
            if (parsed.state) {
                state = core.normalizeState(parsed.state);
            }
            persistConfig();
            persistState();
            rebuildPanel();
        }

        savePresetButton.addEventListener("click", () => {
            const name = (window.prompt("请输入预设名称：", "我的机位预设") || "").trim();
            if (!name) {
                return;
            }

            const presets = getPresets();
            presets[name] = serializeConfiguration();
            writeStorage(storageKey(target.name, "presets"), presets);
            refreshPresetSelect(name);
            showToast(`已保存预设：${name}`);
        });

        presetSelect.addEventListener("change", () => {
            const name = presetSelect.value;
            if (!name) {
                return;
            }

            const preset = getPresets()[name];
            if (!preset) {
                showToast("找不到该预设", "error");
                return;
            }

            try {
                importConfiguration(preset);
                showToast(`已加载预设：${name}`);
            } catch (error) {
                showToast(`预设加载失败：${error.message}`, "error");
            }
        });

        deletePresetButton.addEventListener("click", () => {
            const name = presetSelect.value;
            if (!name) {
                showToast("请先选择一个预设", "warning");
                return;
            }
            if (!window.confirm(`确定删除预设“${name}”吗？`)) {
                return;
            }

            const presets = getPresets();
            delete presets[name];
            writeStorage(storageKey(target.name, "presets"), presets);
            refreshPresetSelect();
            showToast(`已删除预设：${name}`);
        });

        exportButton.addEventListener("click", async () => {
            const copied = await copyText(
                JSON.stringify(serializeConfiguration(), null, 2),
            );
            showToast(copied ? "配置 JSON 已复制" : "复制失败", copied ? "info" : "error");
        });

        importButton.addEventListener("click", async () => {
            const raw = await readClipboardOrPrompt();
            if (!raw.trim()) {
                return;
            }
            try {
                importConfiguration(raw);
                showToast("配置已导入");
            } catch (error) {
                showToast(`配置导入失败：${error.message}`, "error");
            }
        });

        resetConfigButton.addEventListener("click", () => {
            if (!window.confirm("确定恢复原插件的默认机位词与权重吗？")) {
                return;
            }
            config = core.cloneDefaultConfig();
            persistConfig();
            rebuildPanel();
        });

        copyOutputButton.addEventListener("click", async () => {
            const copied = await copyText(output.value);
            showToast(copied ? "机位提示词已复制" : "复制失败", copied ? "info" : "error");
        });

        function resetPosition() {
            state.pos_x = 0;
            state.pos_y = 0;
            state.pos_z = 0;
            state.roll = 0;
            scheduleRender();
        }

        resetPositionButton.addEventListener("click", resetPosition);
        visualResetButton.addEventListener("click", resetPosition);

        dragModeButton.addEventListener("click", () => {
            state.drag_mode =
                state.drag_mode === "relative" ? "absolute" : "relative";
            dragModeButton.textContent =
                state.drag_mode === "absolute" ? "绝对拖拽" : "相对拖拽";
            canvasHint.textContent =
                state.drag_mode === "absolute"
                    ? "拖拽画布 · 鼠标位置即参数"
                    : "拖拽画布 · 抓住世界旋转";
            scheduleRender();
        });

        mergeOnGenerateCheckbox.addEventListener("change", () => {
            uiState.mergeOnGenerate = mergeOnGenerateCheckbox.checked;
            persistUiState();
            showToast(
                uiState.mergeOnGenerate
                    ? "生成时将在任务参数中融合机位提示词"
                    : "已暂停融合；正向提示词保持不变",
            );
        });

        panel.addEventListener("toggle", () => {
            uiState.open = panel.open;
            persistUiState();
            if (panel.open) {
                scheduleRender();
            }
        });

        const context = canvas.getContext("2d");
        context.scale(pixelRatio, pixelRatio);
        const targetHeight = 0.7;
        const verticalOffset = 380 * -0.275;
        let isDragging = false;
        let pointerX = CANVAS_WIDTH / 2;
        let pointerY = CANVAS_HEIGHT / 2;
        let dragStartX = 0;
        let dragStartY = 0;
        let startPosX = 0;
        let startPosY = 0;

        function cameraPosition() {
            const radius = 1.7 - 0.75 * state.pos_z;
            const azimuth = state.pos_x * Math.PI;
            const elevation = core.clamp(
                state.pos_y * Math.PI / 2,
                -Math.PI / 2,
                Math.PI / 2,
            );
            return {
                x: radius * Math.cos(elevation) * Math.sin(azimuth),
                y: targetHeight + radius * Math.sin(elevation),
                z: radius * Math.cos(elevation) * Math.cos(azimuth),
                radius,
                azimuth,
                elevation,
            };
        }

        function project(x, y, z) {
            const depth = 4 - z;
            if (depth < 0.1) {
                return null;
            }
            return {
                x: 280 + (x / depth) * 380,
                y: 250 - ((y - 1.8) / depth) * 380 + verticalOffset,
                depth,
            };
        }

        function roundedRect(x, y, width, height, radius) {
            context.beginPath();
            context.moveTo(x + radius, y);
            context.lineTo(x + width - radius, y);
            context.quadraticCurveTo(x + width, y, x + width, y + radius);
            context.lineTo(x + width, y + height - radius);
            context.quadraticCurveTo(
                x + width,
                y + height,
                x + width - radius,
                y + height,
            );
            context.lineTo(x + radius, y + height);
            context.quadraticCurveTo(x, y + height, x, y + height - radius);
            context.lineTo(x, y + radius);
            context.quadraticCurveTo(x, y, x + radius, y);
            context.closePath();
        }

        function strokeProjectedLine(from, to, color, width = 1.5) {
            if (!from || !to) {
                return;
            }
            context.strokeStyle = color;
            context.lineWidth = width;
            context.beginPath();
            context.moveTo(from.x, from.y);
            context.lineTo(to.x, to.y);
            context.stroke();
        }

        function drawOrbit(backHalf) {
            const camera = cameraPosition();
            const radius = camera.radius;
            const segments = 50;

            function drawSegmentedPath(pointAt, isBack, color) {
                context.beginPath();
                let drawing = false;
                for (let index = 0; index <= segments; index += 1) {
                    const point = pointAt(index / segments);
                    const projected = project(point.x, point.y, point.z);
                    const belongs = isBack(point);

                    if (projected && (backHalf ? belongs : !belongs)) {
                        if (drawing) {
                            context.lineTo(projected.x, projected.y);
                        } else {
                            context.moveTo(projected.x, projected.y);
                            drawing = true;
                        }
                    } else {
                        drawing = false;
                    }
                }
                context.strokeStyle = color;
                context.lineWidth = 2.2;
                context.setLineDash([5, 7]);
                context.stroke();
                context.setLineDash([]);
            }

            drawSegmentedPath(
                (fraction) => {
                    const angle = fraction * Math.PI * 2;
                    return {
                        x: radius * Math.cos(angle),
                        y: targetHeight,
                        z: radius * Math.sin(angle),
                    };
                },
                (point) => point.z <= 0,
                "rgba(77,208,225,0.42)",
            );

            drawSegmentedPath(
                (fraction) => {
                    const angle = -Math.PI / 2 + fraction * Math.PI;
                    return {
                        x: radius * Math.cos(angle) * Math.sin(camera.azimuth),
                        y: targetHeight + radius * Math.sin(angle),
                        z: radius * Math.cos(angle) * Math.cos(camera.azimuth),
                    };
                },
                (point) => point.z <= 0,
                isDragging
                    ? "rgba(255,138,76,0.72)"
                    : "rgba(255,138,76,0.48)",
            );

            drawSegmentedPath(
                (fraction) => {
                    const angle = fraction * Math.PI * 2;
                    return {
                        x: radius * Math.cos(camera.elevation) * Math.cos(angle),
                        y: targetHeight + radius * Math.sin(camera.elevation),
                        z: radius * Math.cos(camera.elevation) * Math.sin(angle),
                    };
                },
                (point) => point.z <= 0,
                isDragging
                    ? "rgba(77,208,225,0.68)"
                    : "rgba(77,208,225,0.40)",
            );
        }

        function drawDirectionMarker(x, y, z, label) {
            const point = project(x, y, z);
            if (!point) {
                return;
            }
            const radius = core.clamp((2.5 / point.depth) * 9, 3.5, 16);
            const gradient = context.createRadialGradient(
                point.x - 0.35 * radius,
                point.y - 0.35 * radius,
                0.2 * radius,
                point.x,
                point.y,
                radius,
            );
            gradient.addColorStop(0, "#ffffff");
            gradient.addColorStop(0.4, "rgba(77,208,225,1)");
            gradient.addColorStop(1, "rgba(0,0,0,0.35)");
            context.fillStyle = gradient;
            context.beginPath();
            context.arc(point.x, point.y, radius, 0, 2 * Math.PI);
            context.fill();

            if (label) {
                context.fillStyle = "rgba(255,255,255,0.94)";
                context.font = `700 ${Math.max(14, radius + 6)}px monospace`;
                context.textAlign = "center";
                context.textBaseline = "middle";
                context.fillText(label, point.x, point.y - radius - 9);
            }
        }

        function drawSubjectAndAxes() {
            const center = project(0, targetHeight, 0);
            if (!center) {
                return;
            }

            const up = project(0, 1.25, 0);
            const down = project(0, targetHeight - 0.55, 0);
            const left = project(0.55, targetHeight, 0);
            const right = project(-0.55, targetHeight, 0);
            const front = project(0, targetHeight, 0.55);
            const back = project(0, targetHeight, -0.55);
            strokeProjectedLine(center, up, "rgba(120,220,255,0.95)", 1.6);
            strokeProjectedLine(center, down, "rgba(120,220,255,0.45)", 1.2);
            strokeProjectedLine(center, left, "rgba(190,190,210,0.8)", 1.2);
            strokeProjectedLine(center, right, "rgba(190,190,210,0.8)", 1.2);
            strokeProjectedLine(center, back, "rgba(190,190,210,0.8)", 1.2);
            strokeProjectedLine(center, front, "#ffd166", 2.4);

            context.fillStyle = "rgba(0,0,0,0.42)";
            context.beginPath();
            context.ellipse(center.x, center.y + 40, 20, 6, 0, 0, 2 * Math.PI);
            context.fill();

            const gradient = context.createRadialGradient(
                center.x - 5,
                center.y - 6,
                3,
                center.x,
                center.y,
                16,
            );
            gradient.addColorStop(0, "#9af0fa");
            gradient.addColorStop(1, "#2f8a93");
            context.fillStyle = gradient;
            context.beginPath();
            context.arc(center.x, center.y, 16, 0, 2 * Math.PI);
            context.fill();
            context.strokeStyle = "rgba(255,255,255,0.25)";
            context.lineWidth = 1;
            context.stroke();
        }

        function drawCamera() {
            const camera = cameraPosition();
            const cameraPoint = project(camera.x, camera.y, camera.z);
            const inwardPoint = project(
                0.9 * camera.x,
                0.9 * camera.y + 0.001,
                0.9 * camera.z,
            );
            const center = project(0, targetHeight, 0);
            if (!cameraPoint || !inwardPoint || !center) {
                return;
            }

            context.strokeStyle = "rgba(255,209,102,0.72)";
            context.lineWidth = 2.5;
            context.setLineDash([5, 7]);
            context.beginPath();
            context.moveTo(cameraPoint.x, cameraPoint.y);
            context.lineTo(center.x, center.y);
            context.stroke();
            context.setLineDash([]);

            const angle = Math.atan2(
                cameraPoint.y - inwardPoint.y,
                cameraPoint.x - inwardPoint.x,
            );
            const distanceScale = 1.4 - 0.45 * state.pos_z;
            const depthScale = core.clamp(2.5 / cameraPoint.depth, 0.5, 2);
            const size = Math.max(8, 14 * distanceScale * depthScale);
            const glowRadius = 2.5 * size;
            const glow = context.createRadialGradient(
                cameraPoint.x,
                cameraPoint.y,
                0,
                cameraPoint.x,
                cameraPoint.y,
                glowRadius,
            );
            glow.addColorStop(0, "rgba(255,138,76,0.55)");
            glow.addColorStop(0.4, "rgba(255,138,76,0.18)");
            glow.addColorStop(1, "rgba(255,138,76,0)");
            context.fillStyle = glow;
            context.fillRect(
                cameraPoint.x - glowRadius,
                cameraPoint.y - glowRadius,
                2 * glowRadius,
                2 * glowRadius,
            );

            if (isDragging) {
                context.strokeStyle = "rgba(255,255,255,0.72)";
                context.lineWidth = 1.5;
                context.beginPath();
                context.arc(
                    cameraPoint.x,
                    cameraPoint.y,
                    1.7 * size,
                    0,
                    2 * Math.PI,
                );
                context.stroke();
            }

            context.save();
            context.translate(cameraPoint.x, cameraPoint.y);
            context.rotate(angle + state.roll * Math.PI / 4);
            const bodyWidth = 1.8 * size;
            const bodyHeight = 1.2 * size;

            context.fillStyle = "rgba(0,0,0,0.5)";
            roundedRect(
                -bodyWidth / 2 + 1,
                -bodyHeight / 2 + 2,
                bodyWidth,
                bodyHeight,
                3,
            );
            context.fill();

            const bodyGradient = context.createLinearGradient(
                0,
                -bodyHeight / 2,
                0,
                bodyHeight / 2,
            );
            bodyGradient.addColorStop(0, "#ffa66a");
            bodyGradient.addColorStop(1, "#e06a30");
            context.fillStyle = bodyGradient;
            roundedRect(
                -bodyWidth / 2,
                -bodyHeight / 2,
                bodyWidth,
                bodyHeight,
                3,
            );
            context.fill();

            context.fillStyle = "#0b0e13";
            context.beginPath();
            context.arc(0.4 * size, 0, 0.42 * size, 0, 2 * Math.PI);
            context.fill();
            context.strokeStyle = "rgba(255,255,255,0.7)";
            context.lineWidth = 1;
            context.stroke();
            context.fillStyle = "rgba(255,255,255,0.35)";
            context.beginPath();
            context.arc(0.35 * size, -0.1 * size, 0.12 * size, 0, 2 * Math.PI);
            context.fill();
            context.restore();
        }

        function drawCanvas() {
            context.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
            const background = context.createLinearGradient(
                0,
                0,
                0,
                CANVAS_HEIGHT,
            );
            background.addColorStop(0, "#080a0f");
            background.addColorStop(1, "#10141d");
            context.fillStyle = background;
            context.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

            const glow = context.createRadialGradient(280, 225, 10, 280, 225, 230);
            glow.addColorStop(0, "rgba(77,208,225,0.07)");
            glow.addColorStop(1, "rgba(77,208,225,0)");
            context.fillStyle = glow;
            context.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

            const camera = cameraPosition();
            drawOrbit(true);
            if (camera.z < 0) {
                drawCamera();
            }
            drawSubjectAndAxes();
            drawOrbit(false);

            const horizontalRadius = camera.radius * Math.cos(camera.elevation);
            const elevationY =
                targetHeight + camera.radius * Math.sin(camera.elevation);
            drawDirectionMarker(0, targetHeight, camera.radius, "F");
            drawDirectionMarker(0, targetHeight, -camera.radius, "B");
            drawDirectionMarker(camera.radius, targetHeight, 0, "L");
            drawDirectionMarker(-camera.radius, targetHeight, 0, "R");
            if (horizontalRadius > 0.05) {
                drawDirectionMarker(0, elevationY, horizontalRadius, "");
                drawDirectionMarker(0, elevationY, -horizontalRadius, "");
                drawDirectionMarker(horizontalRadius, elevationY, 0, "");
                drawDirectionMarker(-horizontalRadius, elevationY, 0, "");
            }

            if (camera.z >= 0) {
                drawCamera();
            }

            if (isDragging) {
                context.strokeStyle = "rgba(255,255,255,0.4)";
                context.lineWidth = 2;
                context.setLineDash([5, 7]);
                context.beginPath();
                context.moveTo(pointerX, 0);
                context.lineTo(pointerX, CANVAS_HEIGHT);
                context.moveTo(0, pointerY);
                context.lineTo(CANVAS_WIDTH, pointerY);
                context.stroke();
                context.setLineDash([]);

                const readout = `X:${state.pos_x.toFixed(2)}  Y:${state.pos_y.toFixed(2)}`;
                context.font = "600 10.5px monospace";
                const width = context.measureText(readout).width;
                let x = pointerX + 14;
                let y = pointerY - 22;
                if (x + width + 12 > CANVAS_WIDTH) {
                    x = pointerX - width - 20;
                }
                if (y < 0) {
                    y = pointerY + 14;
                }
                context.fillStyle = "rgba(0,0,0,0.82)";
                roundedRect(x, y, width + 12, 18, 4);
                context.fill();
                context.fillStyle = "#ff8a4c";
                context.textAlign = "left";
                context.textBaseline = "middle";
                context.fillText(readout, x + 6, y + 9);
            }
        }

        function canvasPoint(event) {
            const rect = canvas.getBoundingClientRect();
            return {
                x: (event.clientX - rect.left) * (CANVAS_WIDTH / rect.width),
                y: (event.clientY - rect.top) * (CANVAS_HEIGHT / rect.height),
            };
        }

        function applyAbsolutePointer(x, y) {
            state.pos_x = core.clamp((x - 280) / 280, -1, 1);
            state.pos_y = core.clamp(-(y - 250) / 250, -1, 1);
        }

        function finishDrag() {
            if (!isDragging) {
                return;
            }
            isDragging = false;
            canvas.style.cursor = "crosshair";
            canvasHint.textContent =
                state.drag_mode === "absolute"
                    ? "拖拽画布 · 鼠标位置即参数"
                    : "拖拽画布 · 抓住世界旋转";
            scheduleRender();
        }

        canvas.addEventListener("pointerdown", (event) => {
            canvas.focus({preventScroll: true});
            canvas.setPointerCapture(event.pointerId);
            const point = canvasPoint(event);
            pointerX = point.x;
            pointerY = point.y;
            dragStartX = point.x;
            dragStartY = point.y;
            startPosX = state.pos_x;
            startPosY = state.pos_y;
            isDragging = true;
            canvas.style.cursor = "none";

            if (state.drag_mode === "absolute") {
                applyAbsolutePointer(point.x, point.y);
            }
            canvasHint.textContent =
                state.drag_mode === "absolute"
                    ? "绝对映射中 · 鼠标即相机方位"
                    : "相对拖拽中 · 抓住世界旋转";
            scheduleRender();
            event.preventDefault();
        });

        canvas.addEventListener("pointermove", (event) => {
            if (!isDragging) {
                return;
            }
            const point = canvasPoint(event);
            pointerX = point.x;
            pointerY = point.y;

            if (state.drag_mode === "absolute") {
                applyAbsolutePointer(point.x, point.y);
            } else {
                let wrapped = startPosX + (point.x - dragStartX) / 280;
                wrapped = ((((wrapped + 1) % 2) + 2) % 2) - 1;
                state.pos_x = wrapped;
                state.pos_y = core.clamp(
                    startPosY - (point.y - dragStartY) / 250,
                    -1,
                    1,
                );
            }
            scheduleRender();
            event.preventDefault();
        });

        canvas.addEventListener("pointerup", finishDrag);
        canvas.addEventListener("pointercancel", finishDrag);

        canvas.addEventListener(
            "wheel",
            (event) => {
                event.preventDefault();
                const delta = normalizedWheelDelta(event);
                if (event.shiftKey) {
                    state.roll = core.clamp(
                        state.roll - 0.002 * delta,
                        -1,
                        1,
                    );
                } else {
                    state.pos_z = core.clamp(
                        state.pos_z -
                            delta * numericValue(config.wheel_step, 0.0003),
                        -1,
                        1,
                    );
                }
                scheduleRender();
            },
            {passive: false},
        );

        canvas.addEventListener("keydown", (event) => {
            const step = event.shiftKey ? 0.1 : 0.02;
            let handled = true;

            switch (event.code) {
                case "ArrowLeft":
                    state.pos_x = core.clamp(state.pos_x - step, -1, 1);
                    break;
                case "ArrowRight":
                    state.pos_x = core.clamp(state.pos_x + step, -1, 1);
                    break;
                case "ArrowUp":
                    state.pos_y = core.clamp(state.pos_y + step, -1, 1);
                    break;
                case "ArrowDown":
                    state.pos_y = core.clamp(state.pos_y - step, -1, 1);
                    break;
                case "BracketLeft":
                    state.pos_z = core.clamp(state.pos_z - step, -1, 1);
                    break;
                case "BracketRight":
                    state.pos_z = core.clamp(state.pos_z + step, -1, 1);
                    break;
                case "Comma":
                    state.roll = core.clamp(state.roll - step, -1, 1);
                    break;
                case "Period":
                    state.roll = core.clamp(state.roll + step, -1, 1);
                    break;
                default:
                    handled = false;
            }

            if (handled) {
                event.preventDefault();
                scheduleRender();
            }
        });

        function render() {
            renderFrame = null;
            currentOutput = core.computeFromState(state, config);
            output.value = currentOutput;
            summaryOutput.textContent = currentOutput || "无输出";
            summaryOutput.title = currentOutput;

            for (const [key, binding] of Object.entries(axisBindings)) {
                binding.slider.value = String(state[key]);
                if (document.activeElement !== binding.number) {
                    binding.number.value = state[key].toFixed(2);
                }
            }

            const degrees = state.pos_x * 180;
            if (Math.abs(state.pos_x) > 0.85) {
                directionReadout.textContent = "BEHIND · 180°";
            } else if (state.pos_x < -0.05) {
                directionReadout.textContent = `LEFT · ${Math.abs(degrees).toFixed(0)}°`;
            } else if (state.pos_x > 0.05) {
                directionReadout.textContent = `RIGHT · ${degrees.toFixed(0)}°`;
            } else {
                directionReadout.textContent = "FRONT · 0°";
            }
            rollReadout.textContent = `ROLL · ${(45 * state.roll).toFixed(0)}°`;

            drawCanvas();
            persistState();
        }

        function scheduleRender() {
            if (renderFrame !== null) {
                return;
            }
            renderFrame = window.requestAnimationFrame(render);
        }

        const promptGroup = getAppRoot().querySelector(
            `#${target.name}_prompt_container`,
        );
        const topRow = getAppRoot().querySelector(`#${target.name}_toprow`);
        const anchor =
            topRow ||
            promptGroup ||
            (promptContainer.tagName === "TEXTAREA"
                ? promptContainer.parentElement
                : promptContainer);
        anchor.insertAdjacentElement("afterend", panel);
        scheduleRender();
    }

    function mountAll() {
        for (const target of TARGETS) {
            mountTarget(target);
        }
    }

    if (typeof window.onUiLoaded === "function") {
        window.onUiLoaded(mountAll);
    } else if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", mountAll, {once: true});
    } else {
        mountAll();
    }

    if (typeof window.onAfterUiUpdate === "function") {
        window.onAfterUiUpdate(mountAll);
    } else {
        const observer = new MutationObserver(mountAll);
        observer.observe(document.documentElement, {childList: true, subtree: true});
    }

    console.info(`[BSK Camera] Forge Neo camera control v${core.VERSION} loaded.`);
})(0);
