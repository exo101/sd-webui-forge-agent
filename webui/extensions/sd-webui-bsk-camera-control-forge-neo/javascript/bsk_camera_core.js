// SPDX-FileCopyrightText: 2026 灰暗x
// SPDX-FileCopyrightText: 2026 Forge Neo port contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Camera prompt calculation ported from ComfyUI-bsk_UI's CameraControlNode.
// Original project: https://github.com/ikusag-png/ComfyUI_bsk_UI

(function (root, factory) {
    "use strict";

    const api = factory();

    if (typeof module === "object" && module.exports) {
        module.exports = api;
    }

    if (root) {
        root.BskForgeCameraCore = api;
    }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    const VERSION = "1.3.0";
    const AZIMUTH_POLE = 0.9;
    const ELEVATION_EYE_MAX = 0.2;
    const LEGACY_KEYS = new Set(["two_tier", "axes", "negative"]);

    const DEFAULT_CONFIG = {
        weight_min: 0.1,
        weight_max: 10.0,
        no_weight: false,
        no_weight_threshold: 0.5,
        azimuth: {
            enabled: true,
            weight: 10.0,
            deadzone_ratio: 0.2,
            directions: {
                front: {tag: "from front", enabled: true},
                back: {tag: "from behind", enabled: true},
                left: {tag: "from right", enabled: true},
                right: {tag: "from left", enabled: true},
            },
        },
        elevation: {
            enabled: true,
            extra: 10.0,
            eye_peak: 3.0,
            categories: {
                bird: {tag: "directly above, from above, aerial view", enabled: true},
                high: {tag: "high angle, from above", enabled: true},
                eye: {tag: "eye-level", enabled: true},
                low: {tag: "low angle, from below", enabled: true},
                worm: {tag: "directly below", enabled: true},
            },
        },
        distance: {
            enabled: true,
            extra: 0.0,
            categories: {
                ecu: {tag: "extreme close-up", enabled: true},
                cu: {tag: "close-up", enabled: true},
                medium: {tag: "medium shot", enabled: true},
                full: {tag: "full body", enabled: true},
                wide: {tag: "wide shot", enabled: true},
            },
        },
        tilt: {
            enabled: true,
            deadzone: 0.15,
            extra: 0.0,
            dutch_tag: "dutch angle",
        },
        extra_master: 1.0,
        wheel_step: 0.0003,
        extras: {
            lens: {enabled: false, value: "85mm lens"},
            dof: {enabled: false, value: "shallow depth of field", weight: 1.3},
            movement: {enabled: false, value: "handheld camera"},
            composition: {enabled: false, value: "rule of thirds"},
            style: {enabled: false, value: "cinematic"},
        },
    };

    const DEFAULT_STATE = {
        pos_x: 0.0,
        pos_y: 0.0,
        pos_z: 0.0,
        roll: 0.0,
        drag_mode: "relative",
    };

    const DISTANCE_RANGES = {
        ecu: [0.7, 1.0],
        cu: [0.2, 0.7],
        medium: [-0.2, 0.2],
        full: [-0.7, -0.2],
        wide: [-1.0, -0.7],
    };

    const DISTANCE_FAR_STRONGER = new Set(["medium", "full", "wide"]);

    function clone(value) {
        return JSON.parse(JSON.stringify(value));
    }

    function isPlainObject(value) {
        return value !== null && typeof value === "object" && !Array.isArray(value);
    }

    function mergeDefaults(target, defaults) {
        const output = isPlainObject(target) ? target : {};

        for (const [key, defaultValue] of Object.entries(defaults)) {
            if (!(key in output)) {
                output[key] = clone(defaultValue);
            } else if (isPlainObject(defaultValue) && isPlainObject(output[key])) {
                mergeDefaults(output[key], defaultValue);
            }
        }

        return output;
    }

    function hasLegacyKeys(value) {
        if (!isPlainObject(value)) {
            return false;
        }

        return Object.entries(value).some(
            ([key, child]) => LEGACY_KEYS.has(key) || hasLegacyKeys(child),
        );
    }

    function stripLegacyKeys(value) {
        if (!isPlainObject(value)) {
            return value;
        }

        for (const key of Object.keys(value)) {
            if (LEGACY_KEYS.has(key)) {
                delete value[key];
            } else {
                stripLegacyKeys(value[key]);
            }
        }

        return value;
    }

    function loadConfig(raw) {
        let config;

        if (!raw) {
            return clone(DEFAULT_CONFIG);
        }

        try {
            config = typeof raw === "string" ? JSON.parse(raw) : clone(raw);
        } catch (_error) {
            return clone(DEFAULT_CONFIG);
        }

        if (!isPlainObject(config)) {
            return clone(DEFAULT_CONFIG);
        }

        if (hasLegacyKeys(config)) {
            config.weight_min = DEFAULT_CONFIG.weight_min;
            config.weight_max = DEFAULT_CONFIG.weight_max;
            stripLegacyKeys(config);
        }

        return mergeDefaults(config, clone(DEFAULT_CONFIG));
    }

    function finiteNumber(value, fallback) {
        const parsed = Number.parseFloat(value);
        return Number.isFinite(parsed) ? parsed : fallback;
    }

    function clamp(value, minimum, maximum) {
        return Math.min(maximum, Math.max(minimum, value));
    }

    function normalizeState(raw) {
        const input = isPlainObject(raw) ? raw : {};
        return {
            pos_x: clamp(finiteNumber(input.pos_x, DEFAULT_STATE.pos_x), -1, 1),
            pos_y: clamp(finiteNumber(input.pos_y, DEFAULT_STATE.pos_y), -1, 1),
            pos_z: clamp(finiteNumber(input.pos_z, DEFAULT_STATE.pos_z), -1, 1),
            roll: clamp(finiteNumber(input.roll, DEFAULT_STATE.roll), -1, 1),
            drag_mode: input.drag_mode === "absolute" ? "absolute" : "relative",
        };
    }

    function formatWeight(value) {
        const rounded = Math.round(finiteNumber(value, 0) * 100) / 100;
        return rounded.toFixed(2);
    }

    function splitTags(tag) {
        return String(tag || "")
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean);
    }

    function emitWeighted(tag, weight) {
        return splitTags(tag).map((item) => `(${item}:${formatWeight(weight)})`);
    }

    function elevationKey(y) {
        if (y > 0.7) {
            return "bird";
        }
        if (y > ELEVATION_EYE_MAX) {
            return "high";
        }
        if (y >= 0) {
            return "eye";
        }
        if (y >= -0.7) {
            return "low";
        }
        return "worm";
    }

    function elevationWeight(config, y, key) {
        if (key === "eye") {
            const peak = finiteNumber(config.elevation.eye_peak, 3);
            const fraction = clamp(y / ELEVATION_EYE_MAX, 0, 1);
            return peak + (1 - peak) * fraction;
        }

        const extraMaster = finiteNumber(config.extra_master, 1);
        const extra = finiteNumber(config.elevation.extra, 0);
        const scale = 1 + extraMaster * extra;

        return scale <= 0 ? 0 : Math.abs(y) * scale;
    }

    function distanceKey(z) {
        if (z > 0.7) {
            return "ecu";
        }
        if (z > 0.2) {
            return "cu";
        }
        if (z >= -0.2) {
            return "medium";
        }
        if (z >= -0.7) {
            return "full";
        }
        return "wide";
    }

    function directionRatios(posX) {
        const azimuth = finiteNumber(posX, 0) * Math.PI;
        let front = Math.max(0, Math.cos(azimuth));
        let back = Math.max(0, -Math.cos(azimuth));
        let right = Math.max(0, Math.sin(azimuth));
        let left = Math.max(0, -Math.sin(azimuth));
        const sum = front + back + left + right;

        if (sum > 0) {
            front /= sum;
            back /= sum;
            left /= sum;
            right /= sum;
        }

        return {front, back, left, right};
    }

    function distanceParts(config, z) {
        const key = distanceKey(z);
        const category = config.distance.categories?.[key];

        if (!category || !category.tag || category.enabled === false) {
            return [];
        }

        const extraMaster = finiteNumber(config.extra_master, 1);
        const extra = finiteNumber(config.distance.extra, 0);
        const weightMin = finiteNumber(config.weight_min, 0.1);
        const weightMax = finiteNumber(config.weight_max, 10);
        const [start, end] = DISTANCE_RANGES[key];
        const fraction = clamp(
            DISTANCE_FAR_STRONGER.has(key)
                ? (end - z) / (end - start)
                : (z - start) / (end - start),
            0,
            1,
        );
        const weight = clamp(1 + fraction * extraMaster * extra, weightMin, weightMax);

        return emitWeighted(category.tag, weight);
    }

    function tiltParts(config) {
        if (config.tilt.enabled === false) {
            return [];
        }

        const extraMaster = finiteNumber(config.extra_master, 1);
        const extra = finiteNumber(config.tilt.extra, 0);
        const weightMax = finiteNumber(config.weight_max, 10);
        const weight = clamp(1 + extraMaster * extra, 0.1, weightMax);

        return emitWeighted(config.tilt.dutch_tag, weight);
    }

    function appendExtras(parts, config, useWeights) {
        const extras = config.extras || {};

        for (const key of ["lens", "dof", "movement", "composition", "style"]) {
            const item = extras[key];
            const value = item?.value?.trim();

            if (!item?.enabled || !value) {
                continue;
            }

            if (useWeights && key === "dof") {
                parts.push(`(${value}:${formatWeight(item.weight ?? 1.3)})`);
            } else {
                parts.push(value);
            }
        }
    }

    function computePromptNoWeight(posX, posY, posZ, roll, config) {
        const parts = [];
        const threshold = finiteNumber(config.no_weight_threshold, 0.5);

        if (config.azimuth.enabled !== false) {
            const ratios = directionRatios(posX);
            const gate = clamp(
                (1 - Math.abs(posY)) / (1 - AZIMUTH_POLE),
                0,
                1,
            );

            if (gate > 0) {
                const entries = Object.entries(ratios);
                let dominantName = null;
                let dominantRatio = -1;

                for (const [name, ratio] of entries) {
                    const direction = config.azimuth.directions?.[name];
                    if (direction && direction.enabled !== false && ratio > dominantRatio) {
                        dominantName = name;
                        dominantRatio = ratio;
                    }
                }

                if (dominantName !== null && dominantRatio > 0) {
                    parts.push(
                        ...splitTags(config.azimuth.directions[dominantName].tag),
                    );
                }

                for (const [name, ratio] of entries) {
                    if (name === dominantName) {
                        continue;
                    }

                    const direction = config.azimuth.directions?.[name];
                    if (
                        direction &&
                        direction.enabled !== false &&
                        ratio >= threshold
                    ) {
                        parts.push(...splitTags(direction.tag));
                    }
                }
            }
        }

        if (config.elevation.enabled !== false) {
            const key = elevationKey(posY);
            const category = config.elevation.categories?.[key];

            if (
                key !== "eye" &&
                category?.tag &&
                category.enabled !== false
            ) {
                parts.push(...splitTags(category.tag));
            }
        }

        if (config.distance.enabled !== false) {
            const key = distanceKey(posZ);
            const category = config.distance.categories?.[key];

            if (
                key !== "medium" &&
                category?.tag &&
                category.enabled !== false
            ) {
                parts.push(...splitTags(category.tag));
            }
        }

        if (
            config.tilt.enabled !== false &&
            Math.abs(roll) >= finiteNumber(config.tilt.deadzone, 0.15)
        ) {
            parts.push(...splitTags(config.tilt.dutch_tag));
        }

        appendExtras(parts, config, false);
        const result = parts.join(", ");
        return result ? `${result},` : "";
    }

    function computePrompt(posX, posY, posZ, roll, rawConfig) {
        const config = loadConfig(rawConfig);
        const x = clamp(finiteNumber(posX, 0), -1, 1);
        const y = clamp(finiteNumber(posY, 0), -1, 1);
        const z = clamp(finiteNumber(posZ, 0), -1, 1);
        const tilt = clamp(finiteNumber(roll, 0), -1, 1);

        if (config.no_weight === true) {
            return computePromptNoWeight(x, y, z, tilt, config);
        }

        const parts = [];
        const weightMin = finiteNumber(config.weight_min, 0.1);
        const weightMax = finiteNumber(config.weight_max, 10);
        const deadzone = finiteNumber(config.azimuth.deadzone_ratio, 0.2);

        if (config.azimuth.enabled !== false) {
            const ratios = directionRatios(x);
            const gate = clamp(
                (1 - Math.abs(y)) / (1 - AZIMUTH_POLE),
                0,
                1,
            );
            const budget = finiteNumber(config.azimuth.weight, 1) * gate;

            for (const [name, ratio] of Object.entries(ratios)) {
                const direction = config.azimuth.directions?.[name];
                let weight = ratio * budget;

                if (
                    !direction ||
                    direction.enabled === false ||
                    ratio <= 0 ||
                    weight < deadzone
                ) {
                    continue;
                }

                weight = clamp(weight, weightMin, weightMax);
                parts.push(...emitWeighted(direction.tag, weight));
            }
        }

        if (config.elevation.enabled !== false) {
            const key = elevationKey(y);
            const category = config.elevation.categories?.[key];

            if (category?.tag && category.enabled !== false) {
                let weight = elevationWeight(config, y, key);

                if (weight > 0) {
                    weight = clamp(weight, weightMin, weightMax);
                    parts.push(...emitWeighted(category.tag, weight));
                }
            }
        }

        if (config.distance.enabled !== false) {
            parts.push(...distanceParts(config, z));
        }

        if (
            config.tilt.enabled !== false &&
            Math.abs(tilt) >= finiteNumber(config.tilt.deadzone, 0.15)
        ) {
            parts.push(...tiltParts(config));
        }

        appendExtras(parts, config, true);
        const result = parts.join(", ");
        return result ? `${result},` : "";
    }

    function computeFromState(rawState, rawConfig) {
        const state = normalizeState(rawState);
        return computePrompt(
            state.pos_x,
            state.pos_y,
            state.pos_z,
            state.roll,
            rawConfig,
        );
    }

    return {
        VERSION,
        DEFAULT_CONFIG,
        DEFAULT_STATE,
        cloneDefaultConfig: () => clone(DEFAULT_CONFIG),
        cloneDefaultState: () => clone(DEFAULT_STATE),
        mergeDefaults,
        loadConfig,
        normalizeState,
        clamp,
        formatWeight,
        splitTags,
        ELEVATION_EYE_MAX,
        elevationKey,
        elevationWeight,
        distanceKey,
        directionRatios,
        computePrompt,
        computePromptNoWeight,
        computeFromState,
    };
});
