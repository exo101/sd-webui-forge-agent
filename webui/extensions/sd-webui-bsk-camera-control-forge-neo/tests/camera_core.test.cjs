// SPDX-License-Identifier: AGPL-3.0-or-later

"use strict";

const assert = require("node:assert/strict");
const core = require("../javascript/bsk_camera_core.js");

function test(name, callback) {
    try {
        callback();
        process.stdout.write(`ok - ${name}\n`);
    } catch (error) {
        process.stderr.write(`not ok - ${name}\n`);
        throw error;
    }
}

test("loads current original defaults", () => {
    const config = core.cloneDefaultConfig();
    assert.equal(config.azimuth.directions.left.tag, "from right");
    assert.equal(config.azimuth.directions.right.tag, "from left");
    assert.equal(config.elevation.eye_peak, 3);
    assert.equal(config.no_weight, false);
});

test("upgrades older saved configs without overwriting custom terms", () => {
    const config = core.loadConfig({
        azimuth: {
            directions: {
                left: {tag: "my saved right view", enabled: true},
            },
        },
        elevation: {extra: 4},
    });

    assert.equal(config.azimuth.directions.left.tag, "my saved right view");
    assert.equal(config.azimuth.directions.right.tag, "from left");
    assert.equal(config.elevation.extra, 4);
    assert.equal(config.elevation.eye_peak, 3);
});

test("default weighted output matches the source algorithm", () => {
    assert.equal(
        core.computePrompt(0, 0, 0, 0, core.cloneDefaultConfig()),
        "(from front:10.00), (eye-level:3.00), (medium shot:1.00),",
    );
});

test("left and right positions use the source viewpoint mapping", () => {
    const config = core.cloneDefaultConfig();
    assert.equal(
        core.computePrompt(-0.5, 0, 0, 0, config),
        "(from right:10.00), (eye-level:3.00), (medium shot:1.00),",
    );
    assert.equal(
        core.computePrompt(0.5, 0, 0, 0, config),
        "(from left:10.00), (eye-level:3.00), (medium shot:1.00),",
    );
});

test("no-weight mode omits neutral eye-level and medium-shot categories", () => {
    const config = core.cloneDefaultConfig();
    config.no_weight = true;
    assert.equal(core.computePrompt(0, 0, 0, 0, config), "from front,");
});

test("fully overhead view gates horizontal directions", () => {
    const config = core.cloneDefaultConfig();
    const output = core.computePrompt(0, 1, 0, 0, config);
    assert.equal(
        output,
        "(directly above:10.00), (from above:10.00), (aerial view:10.00), (medium shot:1.00),",
    );
    assert.equal(output.includes("from front"), false);
});

test("custom direction terms are honored", () => {
    const config = core.cloneDefaultConfig();
    config.no_weight = true;
    config.azimuth.directions.left.tag = "custom right view";
    assert.equal(
        core.computePrompt(-0.5, 0, 0, 0, config),
        "custom right view,",
    );
});

test("distance categories and tilt are emitted", () => {
    const config = core.cloneDefaultConfig();
    config.no_weight = true;
    assert.equal(
        core.computePrompt(0, 0, -0.8, 0.2, config),
        "from front, wide shot, dutch angle,",
    );
});

test("enabled extras and DOF weight are emitted", () => {
    const config = core.cloneDefaultConfig();
    config.extras.lens.enabled = true;
    config.extras.dof.enabled = true;
    assert.equal(
        core.computePrompt(0, 0, 0, 0, config),
        "(from front:10.00), (eye-level:3.00), (medium shot:1.00), 85mm lens, (shallow depth of field:1.30),",
    );
});

test("weighted elevation uses the source center-versus-extreme curve", () => {
    const config = core.cloneDefaultConfig();
    config.azimuth.enabled = false;
    config.distance.enabled = false;
    config.tilt.enabled = false;

    assert.equal(core.computePrompt(0, 0, 0, 0, config), "(eye-level:3.00),");
    assert.equal(core.computePrompt(0, 0.1, 0, 0, config), "(eye-level:2.00),");
    assert.equal(core.computePrompt(0, 0.2, 0, 0, config), "(eye-level:1.00),");
    assert.equal(
        core.computePrompt(0, -0.01, 0, 0, config),
        "(low angle:0.11), (from below:0.11),",
    );
});

test("negative elevation never falls back to eye-level", () => {
    const config = core.cloneDefaultConfig();
    config.no_weight = true;
    config.azimuth.enabled = false;
    config.distance.enabled = false;
    config.tilt.enabled = false;

    assert.equal(core.computePrompt(0, 0, 0, 0, config), "");
    assert.equal(
        core.computePrompt(0, -0.01, 0, 0, config),
        "low angle, from below,",
    );
});

test("invalid state values are clamped safely", () => {
    assert.deepEqual(
        core.normalizeState({
            pos_x: 4,
            pos_y: -4,
            pos_z: "bad",
            roll: 2,
            drag_mode: "unexpected",
        }),
        {
            pos_x: 1,
            pos_y: -1,
            pos_z: 0,
            roll: 1,
            drag_mode: "relative",
        },
    );
});

process.stdout.write("All camera core tests passed.\n");
