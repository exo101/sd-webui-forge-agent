(function () {
    const GALLERY_ID = "img2img_gallery";
    const VIEWER_ID = "img2img_compare_viewer";
    const MODE_TABS_ID = "mode_img2img";

    let beforeData = null;
    let afterData = null;
    let refreshTimer = null;
    let galleryObserverReady = false;
    let sourceObserverReady = false;

    // A src is usable if it's a data URI, a Gradio /file= temp URL, or an http(s) URL
    function isUsableSrc(src) {
        if (!src) return false;
        return src.indexOf("data:") === 0 || src.indexOf("/file=") !== -1 || /^https?:\/\//i.test(src);
    }

    function scheduleRefresh() {
        if (refreshTimer !== null) return;
        refreshTimer = setTimeout(function () {
            refreshTimer = null;
            refresh();
        }, 60);
    }

    // ---- Original image (left side) ----

    // Compose the ForgeCanvas background <img> with its drawing <canvas>
    // (sketch/inpaint scribbles) so the "original" matches what the user drew.
    function readForgeCanvas(root) {
        const img = root.querySelector("img.forge-image");
        if (!img) return null;
        const src = img.getAttribute("src");
        const canvas = root.querySelector("canvas.forge-drawing-canvas");
        if (!canvas || canvas.width < 2 || canvas.height < 2) {
            return isUsableSrc(src) ? src : null;
        }
        const w = img.naturalWidth || canvas.width;
        const h = img.naturalHeight || canvas.height;
        if (!w || !h) return isUsableSrc(src) ? src : null;
        try {
            const off = document.createElement("canvas");
            off.width = w;
            off.height = h;
            const ctx = off.getContext("2d");
            if (img.complete && img.naturalWidth > 0) ctx.drawImage(img, 0, 0, w, h);
            ctx.drawImage(canvas, 0, 0, w, h);
            return off.toDataURL("image/png");
        } catch (e) {
            return isUsableSrc(src) ? src : null;
        }
    }

    // Read the file from a native file input (gr.Image "Inpaint upload" mode)
    function readFileInput(root, callback) {
        const input = root.querySelector('input[type="file"]');
        const file = input && input.files && input.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = function () { callback(reader.result); };
        reader.readAsDataURL(file);
    }

    // Find the currently active mode tab's content panel
    function activeSourcePanel() {
        const tabs = document.getElementById(MODE_TABS_ID);
        if (!tabs) return null;
        const selected = tabs.querySelector('button[role="tab"].selected') ||
            tabs.querySelector('button[role="tab"][aria-selected="true"]');
        if (!selected) return null;
        const controls = selected.getAttribute("aria-controls");
        return controls ? document.getElementById(controls) : null;
    }

    // Synchronously extract the original image URL for the active mode
    function readSourceSync() {
        const panel = activeSourcePanel();
        if (!panel) return null;
        const forgeUrl = readForgeCanvas(panel);
        if (forgeUrl) return forgeUrl;
        // fallback: any usable preview img inside the panel (gr.Image preview)
        const imgs = panel.querySelectorAll("img");
        for (const img of imgs) {
            const src = img.getAttribute("src") || img.currentSrc;
            if (isUsableSrc(src)) return src;
        }
        return null;
    }

    function captureSource() {
        const panel = activeSourcePanel();
        let url = readSourceSync();
        if (!url && panel) {
            // gr.Image keeps the file in its input; the preview img may be absent
            readFileInput(panel, function (dataUrl) {
                beforeData = { url: dataUrl };
                refresh();
            });
        }
        if (url !== null && (!beforeData || beforeData.url !== url)) {
            beforeData = { url: url };
        }
    }

    // ---- Generated image (right side) ----

    function galleryImages() {
        const gallery = document.getElementById(GALLERY_ID);
        if (!gallery) return [];
        const imgs = [];
        for (const img of gallery.querySelectorAll("img")) {
            const src = img.getAttribute("src") || img.currentSrc;
            if (isUsableSrc(src)) imgs.push({ img: img, src: src });
        }
        return imgs;
    }

    function readAfterFromGallery() {
        const items = galleryImages();
        if (!items.length) { afterData = null; return; }
        // Prefer the selected gallery item if Gradio marks one, else the first
        const gallery = document.getElementById(GALLERY_ID);
        const selected = gallery.querySelector(".gallery-item.selected img, .gallery-item[aria-selected=\"true\"] img");
        let chosen = items[0];
        if (selected) {
            const src = selected.getAttribute("src") || selected.currentSrc;
            const found = items.find(function (it) { return it.src === src; });
            if (found) chosen = found;
        }
        afterData = { url: chosen.src };
    }

    // ---- Rendering ----

    function render() {
        const viewer = document.getElementById(VIEWER_ID);
        if (!viewer) return;

        if (!beforeData || !afterData) {
            const hint = !beforeData
                ? "未检测到原图：请在左侧模式标签页上传/绘制原图，生成后自动显示对比"
                : "生成图片后，可在此拖动滑块对比原图与生成结果";
            viewer.innerHTML = '<div class="img2img-compare-empty">' + hint + "</div>";
            return;
        }

        viewer.innerHTML =
            '<div class="img2img-compare-stage">' +
            '    <img class="img2img-compare-image" src="' + beforeData.url + '" alt="原图">' +
            '    <div class="img2img-compare-overlay" style="clip-path: inset(0 0 0 50%);">' +
            '        <img class="img2img-compare-image" src="' + afterData.url + '" alt="生成结果">' +
            "    </div>" +
            '    <div class="img2img-compare-divider" style="left: 50%;"><span></span></div>' +
            "</div>" +
            '<input class="img2img-compare-slider" type="range" min="0" max="100" value="50" aria-label="调整图像对比分界线">' +
            '<div class="img2img-compare-labels"><span>原图（左）</span><span>生成结果（右）</span></div>';

        const slider = viewer.querySelector(".img2img-compare-slider");
        const overlay = viewer.querySelector(".img2img-compare-overlay");
        const divider = viewer.querySelector(".img2img-compare-divider");
        slider.addEventListener("input", function () {
            const value = Number(slider.value);
            overlay.style.clipPath = "inset(0 0 0 " + value + "%)";
            divider.style.left = value + "%";
        });
    }

    function installStyles() {
        if (document.getElementById("img2img-compare-styles")) return;
        const style = document.createElement("style");
        style.id = "img2img-compare-styles";
        style.textContent =
            "#img2img_compare_viewer_host { width: 100%; max-width: 100%; box-sizing: border-box; overflow: hidden; }" +
            ".img2img-compare-viewer { width: 100%; max-width: 720px; margin: 12px auto 0; box-sizing: border-box; }" +
            ".img2img-compare-stage {" +
            "    position: relative; width: 100%; max-width: 100%; min-height: 160px;" +
            "    overflow: hidden; background: #111; border-radius: 6px; box-sizing: border-box;" +
            "}" +
            ".img2img-compare-image {" +
            "    display: block; width: 100%; max-width: 100%; height: 260px;" +
            "    object-fit: contain; user-select: none; pointer-events: none;" +
            "}" +
            ".img2img-compare-overlay {" +
            "    position: absolute; inset: 0; overflow: hidden;" +
            "}" +
            ".img2img-compare-overlay .img2img-compare-image { width: 100%; max-width: none; }" +
            ".img2img-compare-divider {" +
            "    position: absolute; top: 0; bottom: 0; width: 2px;" +
            "    transform: translateX(-1px); background: #fff;" +
            "    box-shadow: 0 0 4px #000; pointer-events: none;" +
            "}" +
            ".img2img-compare-divider span {" +
            "    position: absolute; top: 50%; left: 50%; width: 28px; height: 28px;" +
            "    border: 2px solid #fff; border-radius: 50%;" +
            "    transform: translate(-50%, -50%); background: #333;" +
            "}" +
            ".img2img-compare-divider span::before," +
            ".img2img-compare-divider span::after {" +
            '    content: ""; position: absolute; top: 7px; width: 7px; height: 7px;' +
            "    border-top: 2px solid #fff; border-right: 2px solid #fff;" +
            "}" +
            ".img2img-compare-divider span::before { left: 4px; transform: rotate(-135deg); }" +
            ".img2img-compare-divider span::after { right: 4px; transform: rotate(45deg); }" +
            ".img2img-compare-slider { display: block; width: 100%; margin: 12px 0 4px; }" +
            ".img2img-compare-slider { height: 22px; cursor: ew-resize; accent-color: #4ea1ff; }" +
            ".img2img-compare-labels { display: flex; justify-content: space-between; color: #aaa; font-size: 12px; }" +
            ".img2img-compare-empty { padding: 70px 12px; color: #888; text-align: center; background: #111; border-radius: 6px; }";
        document.head.appendChild(style);
    }

    function refresh() {
        const viewer = document.getElementById(VIEWER_ID);
        if (!viewer) return;
        captureSource();
        readAfterFromGallery();
        render();
    }

    // Gallery item click: switch the "after" image to the clicked one
    function installGalleryClick() {
        const gallery = document.getElementById(GALLERY_ID);
        if (!gallery || gallery.dataset.i2iCompareClickBound) return;
        gallery.dataset.i2iCompareClickBound = "1";
        gallery.addEventListener("click", function (event) {
            const item = event.target.closest ? event.target.closest(".gallery-item") : null;
            if (!item) return;
            const img = item.querySelector("img");
            if (!img) return;
            const src = img.getAttribute("src") || img.currentSrc;
            if (!isUsableSrc(src)) return;
            afterData = { url: src };
            render();
        });
    }

    function installObservers() {
        if (sourceObserverReady) return;
        sourceObserverReady = true;
        const sources = document.getElementById(MODE_TABS_ID);
        if (sources) {
            new MutationObserver(scheduleRefresh).observe(sources, {
                childList: true,
                subtree: true,
                attributes: true,
                attributeFilter: ["src", "class", "aria-selected"],
            });
        }
        if (galleryObserverReady) return;
        galleryObserverReady = true;
        const gallery = document.getElementById(GALLERY_ID);
        if (gallery) {
            new MutationObserver(scheduleRefresh).observe(gallery, {
                childList: true,
                subtree: true,
                attributes: true,
                attributeFilter: ["src", "class"],
            });
        }
    }

    function setup() {
        installStyles();
        installObservers();
        installGalleryClick();
        const ready = Boolean(
            document.getElementById(GALLERY_ID) &&
            document.getElementById(VIEWER_ID),
        );
        if (!ready) return false;
        refresh();
        return true;
    }

    onUiLoaded(function () {
        if (setup()) return;
        // The img2img tab may render its components lazily; poll until they exist.
        const timer = setInterval(function () {
            if (setup()) clearInterval(timer);
        }, 100);
    });
})();
