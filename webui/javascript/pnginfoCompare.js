(function () {
    const BEFORE_ID = "pnginfo_compare_before";
    const AFTER_ID = "pnginfo_compare_after";
    const VIEWER_ID = "pnginfo_compare_viewer";
    let beforeData = null;
    let afterData = null;

    function readImage(input, callback) {
        const file = input && input.files && input.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = () => callback({
            url: reader.result,
            width: 0,
            height: 0,
        });
        reader.readAsDataURL(file);
    }

    function render() {
        const viewer = document.getElementById(VIEWER_ID);
        if (!viewer) return;

        if (!beforeData || !afterData) {
            viewer.innerHTML = '<div class="pnginfo-compare-empty">请上传两张图片进行对比</div>';
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
            <div class="pnginfo-compare-labels"><span>原图</span><span>对比图</span></div>
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

    function bind() {
        return Boolean(
            document.getElementById(BEFORE_ID) &&
            document.getElementById(AFTER_ID) &&
            document.getElementById(VIEWER_ID),
        );
    }

    function installUploadListener() {
        if (document.body.dataset.pnginfoCompareUploadBound) return;
        document.addEventListener("change", (event) => {
            if (!(event.target instanceof HTMLInputElement) || event.target.type !== "file") return;
            const beforeRoot = document.getElementById(BEFORE_ID);
            const afterRoot = document.getElementById(AFTER_ID);
            if (beforeRoot && beforeRoot.contains(event.target)) {
                readImage(event.target, (data) => {
                    beforeData = data;
                    render();
                });
            } else if (afterRoot && afterRoot.contains(event.target)) {
                readImage(event.target, (data) => {
                    afterData = data;
                    render();
                });
            }
        });
        document.body.dataset.pnginfoCompareUploadBound = "true";
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

    function setup() {
        installStyles();
        installUploadListener();
        if (bind()) return;
        setTimeout(setup, 100);
    }

    onUiLoaded(() => {
        setup();
        new MutationObserver(bind).observe(document.body, {
            childList: true,
            subtree: true,
        });
    });
})();
