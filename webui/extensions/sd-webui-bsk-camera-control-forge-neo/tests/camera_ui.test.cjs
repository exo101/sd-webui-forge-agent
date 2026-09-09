"use strict";

const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const {JSDOM} = require("jsdom");

const extensionRoot = path.resolve(__dirname, "..");
const core = require(path.join(
    extensionRoot,
    "javascript",
    "bsk_camera_core.js",
));
const uiSource = fs.readFileSync(
    path.join(extensionRoot, "javascript", "bsk_camera_ui.js"),
    "utf8",
);

const dom = new JSDOM(
    `<!doctype html>
    <html>
      <body>
        <div id="txt2img_toprow">
          <div id="txt2img_prompt_container">
            <div id="txt2img_prompt"><textarea>masterpiece</textarea></div>
            <div id="txt2img_neg_prompt"><textarea>low quality</textarea></div>
          </div>
          <div id="txt2img_actions_column">
            <button id="txt2img_generate"><span>Generate</span></button>
          </div>
        </div>
        <div id="img2img_toprow">
          <div id="img2img_prompt_container">
            <div id="img2img_prompt"><textarea>portrait</textarea></div>
            <div id="img2img_neg_prompt"><textarea>blurry</textarea></div>
          </div>
          <div id="img2img_actions_column">
            <button id="img2img_generate"><span>Generate</span></button>
          </div>
        </div>
      </body>
    </html>`,
    {
        url: "http://127.0.0.1:7860/",
        pretendToBeVisual: true,
        runScripts: "outside-only",
    },
);

const {window} = dom;
const context = new Proxy(
    {
        createLinearGradient: () => ({addColorStop() {}}),
        createRadialGradient: () => ({addColorStop() {}}),
        measureText: (text) => ({width: String(text).length * 6}),
    },
    {
        get(target, property) {
            if (property in target) {
                return target[property];
            }
            return () => {};
        },
        set(target, property, value) {
            target[property] = value;
            return true;
        },
    },
);

window.HTMLCanvasElement.prototype.getContext = () => context;
window.HTMLCanvasElement.prototype.setPointerCapture = () => {};
window.HTMLCanvasElement.prototype.getBoundingClientRect = () => ({
    left: 0,
    top: 0,
    width: 560,
    height: 500,
    right: 560,
    bottom: 500,
});
window.BskForgeCameraCore = core;
window.gradioApp = () => window.document;
window.confirm = () => true;
window.prompt = () => "";
window.navigator.clipboard = {
    writeText: async () => {},
    readText: async () => "",
};

const loadedCallbacks = [];
const updateCallbacks = [];
window.onUiLoaded = (callback) => loadedCallbacks.push(callback);
window.onAfterUiUpdate = (callback) => updateCallbacks.push(callback);

const forgeCalls = {
    txt2img: [],
    img2img: [],
};

window.submit = function submit() {
    const result = Array.from(arguments);
    result[0] = "txt2img-task-id";
    forgeCalls.txt2img.push(result);
    return result;
};

window.submit_img2img = function submitImg2img() {
    const result = Array.from(arguments);
    result[0] = "img2img-task-id";
    forgeCalls.img2img.push(result);
    return result;
};

window.submit_txt2img_upscale = function submitTxt2imgUpscale() {
    const result = window.submit(...arguments);
    result[2] = 0;
    return result;
};

window.eval(uiSource);
for (const callback of loadedCallbacks) {
    callback();
}

function wait(milliseconds = 20) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function labelByText(panel, text) {
    return [...panel.querySelectorAll("label")].find(
        (label) => label.textContent.trim() === text,
    );
}

(async () => {
    await wait(30);

    const txtPanel = window.document.getElementById(
        "bsk-camera-forge-neo-txt2img",
    );
    const imgPanel = window.document.getElementById(
        "bsk-camera-forge-neo-img2img",
    );
    assert.ok(txtPanel, "txt2img panel should mount");
    assert.ok(imgPanel, "img2img panel should mount");
    assert.equal(
        txtPanel.previousElementSibling.id,
        "txt2img_toprow",
        "panel should mount below the complete Forge prompt/action row",
    );

    const txtPrompt = window.document.querySelector(
        "#txt2img_prompt textarea",
    );
    const txtNegative = window.document.querySelector(
        "#txt2img_neg_prompt textarea",
    );
    const txtGenerate = window.document.getElementById("txt2img_generate");
    const txtOutput = txtPanel.querySelector(".bsk-camera-output");
    assert.equal(txtOutput.readOnly, true);
    assert.equal(
        txtOutput.value,
        "(from front:10.00), (eye-level:3.00), (medium shot:1.00),",
    );
    const eyePeakLabel = labelByText(
        txtPanel,
        "平视权重（正平视时）",
    );
    assert.ok(eyePeakLabel, "the current source eye-level weight control should exist");
    assert.equal(
        eyePeakLabel.parentElement.querySelector("input[type='number']").value,
        "3.00",
    );

    let txtPromptInputEvents = 0;
    txtPrompt.addEventListener("input", () => {
        txtPromptInputEvents += 1;
    });
    txtGenerate.addEventListener("click", () => {
        // Gradio may collect and submit its component values asynchronously.
        // The camera bridge must still inject into submit() arguments without
        // first changing the visible textarea.
        window.setTimeout(() => {
            window.submit(
                null,
                txtPrompt.value,
                txtNegative.value,
                [],
            );
        }, 0);
    });

    const xSlider = txtPanel.querySelector(
        ".bsk-camera-axis-row input[type='range']",
    );
    xSlider.value = "0.5";
    xSlider.dispatchEvent(new window.Event("input", {bubbles: true}));
    await wait(30);
    assert.match(txtOutput.value, /from left/u);
    assert.equal(
        txtPrompt.value,
        "masterpiece",
        "moving the camera must never rewrite the positive prompt",
    );

    const directionFieldset = [...txtPanel.querySelectorAll("fieldset")].find(
        (fieldset) => fieldset.querySelector("legend")?.textContent === "方位",
    );
    const rightRow = [...directionFieldset.querySelectorAll(".text")].find(
        (row) =>
            row.querySelector("label:not(.bsk-camera-checkbox-label)")
                ?.textContent === "右",
    );
    const rightInput = rightRow.querySelector("input[type='text']");
    rightInput.value = "custom left-facing view";
    rightInput.dispatchEvent(new window.Event("input", {bubbles: true}));
    await wait(30);
    assert.match(txtOutput.value, /custom left-facing view/u);
    assert.equal(
        txtPrompt.value,
        "masterpiece",
        "editing a camera term must only update the separate camera box",
    );

    const expectedMerged = `masterpiece, ${txtOutput.value}`;
    txtGenerate.querySelector("span").click();
    await wait();
    assert.equal(
        forgeCalls.txt2img.at(-1)[1],
        expectedMerged,
        "Forge submit arguments should receive the merged prompt",
    );
    assert.equal(
        txtPrompt.value,
        "masterpiece",
        "the visible positive prompt must never be modified during submission",
    );
    assert.equal(
        txtPromptInputEvents,
        0,
        "generation must not dispatch synthetic prompt input events",
    );

    txtGenerate.click();
    await wait();
    txtGenerate.click();
    await wait();
    assert.deepEqual(
        forgeCalls.txt2img.slice(-2).map((args) => args[1]),
        [expectedMerged, expectedMerged],
        "sequential generations must not accumulate camera prompts",
    );
    assert.equal(txtPrompt.value, "masterpiece");

    txtGenerate.click();
    txtGenerate.click();
    await wait();
    assert.deepEqual(
        forgeCalls.txt2img.slice(-2).map((args) => args[1]),
        [expectedMerged, expectedMerged],
        "programmatic double clicks in one task must each merge exactly once",
    );
    assert.equal(txtPrompt.value, "masterpiece");

    window.document.addEventListener("keydown", (event) => {
        if (event.ctrlKey && event.key === "Enter") {
            txtGenerate.click();
        }
    });
    window.document.dispatchEvent(
        new window.KeyboardEvent("keydown", {
            key: "Enter",
            ctrlKey: true,
            bubbles: true,
        }),
    );
    await wait();
    assert.equal(
        forgeCalls.txt2img.at(-1)[1],
        expectedMerged,
        "keyboard-triggered programmatic clicks should use the same merge path",
    );
    assert.equal(txtPrompt.value, "masterpiece");

    const mergeLabel = labelByText(
        txtPanel,
        "生成时融合机位提示词",
    );
    const mergeCheckbox = mergeLabel.querySelector("input");
    mergeCheckbox.checked = false;
    mergeCheckbox.dispatchEvent(new window.Event("change", {bubbles: true}));
    txtGenerate.click();
    await wait();
    assert.equal(
        forgeCalls.txt2img.at(-1)[1],
        "masterpiece",
        "disabled merging should submit the untouched positive prompt",
    );
    assert.equal(txtPrompt.value, "masterpiece");

    mergeCheckbox.checked = true;
    mergeCheckbox.dispatchEvent(new window.Event("change", {bubbles: true}));

    const imgPrompt = window.document.querySelector(
        "#img2img_prompt textarea",
    );
    const imgNegative = window.document.querySelector(
        "#img2img_neg_prompt textarea",
    );
    const imgGenerate = window.document.getElementById("img2img_generate");
    const imgOutput = imgPanel.querySelector(".bsk-camera-output");
    imgGenerate.addEventListener("click", () => {
        window.setTimeout(() => {
            window.submit_img2img(
                null,
                0,
                imgPrompt.value,
                imgNegative.value,
                [],
            );
        }, 0);
    });
    imgGenerate.click();
    await wait();
    assert.equal(
        forgeCalls.img2img.at(-1)[2],
        `portrait, ${imgOutput.value}`,
        "img2img should use its own independent camera prompt",
    );
    assert.equal(imgPrompt.value, "portrait");
    assert.equal(txtPrompt.value, "masterpiece");

    const directArgs = [null, "masterpiece", "low quality", []];
    const directResult = window.submit(...directArgs);
    assert.equal(
        directResult[1],
        expectedMerged,
        "prompt Enter and other direct submit() paths should merge camera words",
    );
    assert.equal(
        directArgs[1],
        "masterpiece",
        "the caller's argument array must not be mutated",
    );

    const upscaleGallery = [{name: "sample.png"}];
    const upscaleResult = window.submit_txt2img_upscale(
        null,
        upscaleGallery,
        0,
        "{}",
        "masterpiece",
        "low quality",
    );
    assert.equal(
        upscaleResult[1],
        upscaleGallery,
        "txt2img upscale gallery arguments must not be mistaken for a prompt",
    );
    assert.equal(
        upscaleResult[4],
        "masterpiece",
        "txt2img upscale requests must not receive camera prompt injection",
    );

    assert.equal(txtNegative.value, "low quality");
    assert.equal(
        window.document.querySelector("#img2img_neg_prompt textarea").value,
        "blurry",
        "negative prompts must remain untouched",
    );

    const wrappedSubmit = window.submit;
    const wrappedImg2imgSubmit = window.submit_img2img;
    for (const callback of updateCallbacks) {
        callback();
    }
    assert.equal(
        window.submit,
        wrappedSubmit,
        "UI refresh must not wrap txt2img submit repeatedly",
    );
    assert.equal(
        window.submit_img2img,
        wrappedImg2imgSubmit,
        "UI refresh must not wrap img2img submit repeatedly",
    );
    assert.equal(
        window.document.querySelectorAll(
            "#bsk-camera-forge-neo-txt2img",
        ).length,
        1,
        "UI update callback must not duplicate the panel",
    );

    process.stdout.write("Camera UI generation bridge tests passed.\n");
})().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
