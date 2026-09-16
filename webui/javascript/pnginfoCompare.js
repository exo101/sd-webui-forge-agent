(function () {
    const BEFORE_ID = "pnginfo_compare_before";
    const AFTER_ID = "pnginfo_compare_after";
    const VIEWER_ID = "pnginfo_compare_viewer";
    let beforeData = null;
    let afterData = null;
    let previewObserverReady = false;

    // A src is usable if it's a data URI, a Gradio /file= temp URL, or an http(s) URL
    function isUsableSrc(src) {
        if (!src) return false;
        return src.indexOf("data:") === 0 || src.indexOf("/file=") !== -1 || /^https?:\/\//i.test(src);
    }

    function setSide(side, url) {
        if (!isUsableSrc(url)) return;
        if (side === "before") beforeData = { url };
        else if (side === "after") afterData = { url };
        render();
    }

    function scanPreviewImages(root = document) {
        const beforeRoot = document.getElementById(BEFORE_ID);
        const afterRoot = document.getElementById(AFTER_ID);
        if (!beforeRoot && !afterRoot) return;
        const images = [];
        if (root instanceof HTMLImageElement) images.push(root);
        if (root.querySelectorAll) images.push(...root.querySelectorAll("img"));
        for (const img of images) {
            const src = img.getAttribute("src") || img.currentSrc;
            if (!isUsableSrc(src)) continue;
            if (beforeRoot && beforeRoot.contains(img)) setSide("before", src);
            else if (afterRoot && afterRoot.contains(img)) setSide("after", src);
        }
    }

    function render() {
        const viewer = document.getElementById(VIEWER_ID);
        if (!viewer) return;

        if (!beforeData || !afterData) {
            viewer.innerHTML = '<div class="pnginfo-compare-empty">请上传两张图片进行对比（"原图" 与 "对比图" 各一张，支持点击选择、拖拽或粘贴）</div>';
            return;
        }

        viewer.innerHTML = `
            <div class="pnginfo-compare-stage">
                <img class="pnginfo-compare-image" src="${beforeData.url}" alt="原图">
                <div class="pnginfo-compare-overlay" style="clip-path: inset(0 50% 0 0);">
                    <img class="pnginfo-compare-image" src="${afterData.url}" alt="对比图">
                </div>
                <div class="pnginfo-compare-divider" style="left: 50%;">
                    <span></span>
                </div>
            </div>
            <input class="pnginfo-compare-slider" type="range" min="0" max="100" value="50" aria-label="调整图像对比分界线">
            <div class="pnginfo-compare-labels"><span>原图（左）</span><span>对比图（右）</span></div>
        `;

        const slider = viewer.querySelector(".pnginfo-compare-slider");
        const overlay = viewer.querySelector(".pnginfo-compare-overlay");
        const divider = viewer.querySelector(".pnginfo-compare-divider");
        slider.addEventListener("input", () => {
            const value = Number(slider.value);
            overlay.style.clipPath = `inset(0 ${100 - value}% 0 0)`;
            divider.style.left = `${value}%`;
        });
    }

    // Fast path: read the chosen file directly from the picker input.
    function bindFileInputs() {
        const roots = [
            [document.getElementById(BEFORE_ID), "before"],
            [document.getElementById(AFTER_ID), "after"],
        ];
        for (const [root, side] of roots) {
            if (!root) continue;
            const input = root.querySelector('input[type="file"]');
            if (!input || input.dataset.pnginfoCmpBound) continue;
            input.dataset.pnginfoCmpBound = "1";
            input.addEventListener("change", (e) => {
                const file = e.target.files && e.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = () => setSide(side, reader.result);
                reader.readAsDataURL(file);
            });
        }
    }

    // Gradio 5 may replace the native file input after the component has
    // rendered. Keep a delegated listener on document so replacement inputs,
    // drag/drop and picker uploads all use the same path.
    function installUploadDelegation() {
        if (document.body.dataset.pnginfoCompareUploadDelegated) return;
        document.addEventListener("change", (event) => {
            const input = event.target;
            if (!(input instanceof HTMLInputElement) || input.type !== "file") return;
            const beforeRoot = document.getElementById(BEFORE_ID);
            const afterRoot = document.getElementById(AFTER_ID);
            const side = beforeRoot && beforeRoot.contains(input)
                ? "before"
                : afterRoot && afterRoot.contains(input) ? "after" : null;
            const file = input.files && input.files[0];
            if (!side || !file) return;
            const reader = new FileReader();
            reader.onload = () => setSide(side, reader.result);
            reader.readAsDataURL(file);
        }, true);
        document.body.dataset.pnginfoCompareUploadDelegated = "true";
    }

    // Robust path: watch the preview <img> that Gradio renders after ANY upload
    // (picker, drag & drop, clipboard paste all end up as a preview img with a src).
    function installPreviewObserver() {
        if (previewObserverReady) return;
        previewObserverReady = true;
        new MutationObserver((mutations) => {
            for (const m of mutations) {
                if (m.type === "childList") {
                    bindFileInputs();
                    for (const node of m.addedNodes) scanPreviewImages(node);
                    continue;
                }
                if (m.type !== "attributes" || m.attributeName !== "src") continue;
                const img = m.target;
                if (!(img instanceof HTMLImageElement)) continue;
                const src = img.getAttribute("src");
                if (!isUsableSrc(src)) continue;
                const beforeRoot = document.getElementById(BEFORE_ID);
                const afterRoot = document.getElementById(AFTER_ID);
                if (beforeRoot && beforeRoot.contains(img)) setSide("before", src);
                else if (afterRoot && afterRoot.contains(img)) setSide("after", src);
            }
        }).observe(document.body, {
            attributes: true,
            attributeFilter: ["src"],
            childList: true,
            subtree: true,
        });
    }

    function installStyles() {
        if (document.getElementById("pnginfo-compare-styles")) return;
        const style = document.createElement("style");
        style.id = "pnginfo-compare-styles";
        style.textContent = `
            #pnginfo_compare_viewer_host { width: 100%; }
            .pnginfo-compare-viewer { width: 100%; margin-top: 12px; }
            .pnginfo-compare-stage {
                position: relative; width: 100%; min-height: 280px;
                overflow: hidden; background: #111; border-radius: 6px;
            }
            .pnginfo-compare-image {
                display: block; width: 100%; height: 480px;
                object-fit: contain; user-select: none; pointer-events: none;
            }
            .pnginfo-compare-overlay {
                position: absolute; inset: 0; overflow: hidden;
            }
            .pnginfo-compare-overlay .pnginfo-compare-image { width: 100%; max-width: none; }
            .pnginfo-compare-divider {
                position: absolute; top: 0; bottom: 0; width: 2px;
                transform: translateX(-1px); background: #fff;
                box-shadow: 0 0 4px #000; pointer-events: none;
            }
            .pnginfo-compare-divider span {
                position: absolute; top: 50%; left: 50%; width: 28px; height: 28px;
                border: 2px solid #fff; border-radius: 50%;
                transform: translate(-50%, -50%); background: #333;
            }
            .pnginfo-compare-divider span::before,
            .pnginfo-compare-divider span::after {
                content: ""; position: absolute; top: 7px; width: 7px; height: 7px;
                border-top: 2px solid #fff; border-right: 2px solid #fff;
            }
            .pnginfo-compare-divider span::before { left: 4px; transform: rotate(-135deg); }
            .pnginfo-compare-divider span::after { right: 4px; transform: rotate(45deg); }
            .pnginfo-compare-slider { display: block; width: 100%; margin: 12px 0 4px; }
            .pnginfo-compare-slider { height: 22px; cursor: ew-resize; accent-color: #4ea1ff; }
            .pnginfo-compare-labels { display: flex; justify-content: space-between; color: #aaa; font-size: 12px; }
            .pnginfo-compare-empty { padding: 70px 12px; color: #888; text-align: center; background: #111; border-radius: 6px; }
        `;
        document.head.appendChild(style);
    }

    function uiReady() {
        return Boolean(
            document.getElementById(BEFORE_ID) &&
            document.getElementById(AFTER_ID) &&
            document.getElementById(VIEWER_ID),
        );
    }

    function setup() {
        installStyles();
        installPreviewObserver();
        installUploadDelegation();
        bindFileInputs();
        scanPreviewImages();
        if (!uiReady()) return false;
        render();
        return true;
    }

    onUiLoaded(() => {
        if (setup()) return;
        // The PNG Info tab may render its components lazily; poll until they exist.
        const timer = setInterval(() => {
            if (setup()) clearInterval(timer);
        }, 100);
    });
})();
