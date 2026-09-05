/**
 * Image Comparison — before/after slider.
 *
 * Two images are stacked (absolute inset:0, object-fit:contain).
 * Image A is the base (bottom). Image B lives inside an overlay
 * (overflow:hidden) whose width changes as the user drags a vertical bar.
 * Images themselves NEVER move or resize — only the overlay clips them.
 *
 * Depends on CSS in style.css:
 *   #img_comp_row      { position:relative; overflow:hidden; }
 *   #img_comp_row img  { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }
 *   #img_comp_overlay  { position:absolute; inset:0; overflow:hidden; }
 *   #img_comp_row .bar { position:absolute; ... }
 */
(function() {
    'use strict';

    var initialized = false;

    // ============================================================
    //  DOM restructuring: move Gradio's <img> elements into the
    //  overlay-based comparison structure.
    // ============================================================

    function getImgSrc(elemId) {
        var root = document.getElementById(elemId);
        if (!root) return null;
        // Gradio wraps <img> inside Svelte divs; find the actual <img>
        var img = root.querySelector('img') || root.querySelector('canvas');
        if (img && img.src) return img.src;
        // Some Gradio versions store the URL in a data attribute
        if (root.dataset && root.dataset.src) return root.dataset.src;
        return null;
    }

    function buildComparisonDOM() {
        var row = document.getElementById('img_comp_row');
        if (!row) return false;
        if (row.querySelector('#img_comp_overlay')) return true; // already built

        // Grab the two Gradio image containers — these are DUMMY display placeholders
        // (grey 16x16 images), NOT the user uploads. Move them to a hidden div so they
        // don't show through the comparison slider.
        var aRoot = document.getElementById('img_comp_A');
        var bRoot = document.getElementById('img_comp_B');
        if (!aRoot || !bRoot) return false;

        // Move dummy containers to a hidden div outside the row
        var srcsContainer = document.createElement('div');
        srcsContainer.id = 'img_comp_srcs';
        srcsContainer.style.cssText = 'display:none !important; position:absolute; width:0; height:0; overflow:hidden;';
        row.parentNode.insertBefore(srcsContainer, row);
        srcsContainer.appendChild(aRoot);
        srcsContainer.appendChild(bRoot);

        // Base image (layer A) — starts empty, populated by loadImage() from real uploads
        var baseBlock = document.createElement('div');
        baseBlock.className = 'comp-block';
        baseBlock.id = 'img_comp_base';
        var baseImg = document.createElement('img');
        baseImg.alt = 'Image A';
        baseImg.draggable = false; // 阻止浏览器原生图片拖拽拦截 pointer 事件
        baseImg.style.pointerEvents = 'none'; // 让鼠标事件穿透到 row
        baseBlock.appendChild(baseImg);
        row.appendChild(baseBlock);

        // Overlay wrapping image B — 始终 100% 宽度，由 clip-path 裁切
        var overlay = document.createElement('div');
        overlay.id = 'img_comp_overlay';
        overlay.style.width = '100%';
        overlay.innerHTML =
            '<div class="comp-block" id="img_comp_top" style="clip-path: inset(0 50% 0 0)">' +
                '<img alt="Image B" draggable="false" style="pointer-events:none">' +
            '</div>';
        row.appendChild(overlay);

        // Drag bar — thin visible line
        var bar = document.createElement('div');
        bar.className = 'bar';
        bar.id = 'img_comp_bar';
        bar.style.left = '50%';
        row.appendChild(bar);

        // Drag handle — thin invisible hit area centered on the bar
        var handle = document.createElement('div');
        handle.className = 'comp-handle';
        handle.id = 'img_comp_handle';
        handle.style.left = '50%';
        row.appendChild(handle);

        return true;
    }

    function extractImageSrc(root) {
        if (!root) return null;
        var candidates = root.querySelectorAll('img, canvas');
        for (var i = 0; i < candidates.length; i++) {
            var candidate = candidates[i];
            var src = candidate.currentSrc ||
                candidate.src ||
                candidate.getAttribute('src') ||
                candidate.getAttribute('data-src') ||
                candidate.getAttribute('data-url');
            if (src) return src;
        }
        // Gradio sometimes uses a background-image on a wrapper
        var withBg = root.querySelectorAll('[style*="background-image"]');
        for (var j = 0; j < withBg.length; j++) {
            var background = withBg[j].style.backgroundImage ||
                getComputedStyle(withBg[j]).backgroundImage;
            var m = background && background.match(/url\(["']?(.+?)["']?\)/);
            if (m) return m[1];
        }
        // data-src fallback
        if (root.dataset) {
            return root.dataset.src || root.dataset.url || root.dataset.preview || null;
        }
        return null;
    }

    function applyComparisonSources(srcA, srcB) {
        var baseImg = document.querySelector('#img_comp_base img');
        var topImg = document.querySelector('#img_comp_top img');
        var topBlock = document.getElementById('img_comp_top');
        if (baseImg) {
            baseImg.src = srcA || '';
            baseImg.draggable = false; // 安全网：Gradio 可能替换 img
            baseImg.style.pointerEvents = 'none';
            // 用 !important 压过 CSS 中可能存在的 display:none !important 规则
            baseImg.style.setProperty('display', srcA ? 'block' : 'none', 'important');
        }
        if (topImg) {
            topImg.src = srcB || '';
            topImg.draggable = false;
            topImg.style.pointerEvents = 'none';
            topImg.style.setProperty('display', srcB ? 'block' : 'none', 'important');
        }
        if (topBlock) {
            topBlock.style.setProperty('display', srcB ? 'block' : 'none', 'important');
            topBlock.style.setProperty('visibility', 'visible', 'important');
            topBlock.style.setProperty('opacity', '1', 'important');
        }
    }

    // ============================================================
    //  Drag logic
    // ============================================================

    var dragging = false;
    var didDrag = false;
    var startX = 0;

    function setSplit(xPercent) {
        xPercent = Math.max(0, Math.min(100, xPercent));
        var topBlock = document.getElementById('img_comp_top');
        var overlay = document.getElementById('img_comp_overlay');
        var bar = document.getElementById('img_comp_bar');
        var handle = document.getElementById('img_comp_handle');
        // overlay 始终保持 100% 宽度，避免内部 img 跟随缩放
        if (overlay) overlay.style.width = '100%';
        // 用 clip-path 裁切 Image B 图层（inset: top right bottom left）
        // xPercent=50 表示显示左侧 50%，右侧裁掉 (100-50)%
        if (topBlock) topBlock.style.clipPath = 'inset(0 ' + (100 - xPercent) + '% 0 0)';
        if (bar) bar.style.left = xPercent + '%';
        if (handle) handle.style.left = xPercent + '%';
    }

    function pointerToPercent(clientX) {
        var row = document.getElementById('img_comp_row');
        if (!row) return 50;
        var rect = row.getBoundingClientRect();
        var x = clientX - rect.left;
        return (x / rect.width) * 100;
    }

    function onPointerDown(e) {
        dragging = true;
        didDrag = false;
        startX = e.clientX || 0;
        // Capture pointer so we keep receiving move events outside the row
        if (e.target.setPointerCapture) {
            try { e.target.setPointerCapture(e.pointerId); } catch (_) {}
        }
        e.preventDefault();
    }

    function onPointerMove(e) {
        if (!dragging) return;
        // Only mark as drag if moved > 5px (avoids accidental drag on click)
        if (Math.abs((e.clientX || 0) - startX) > 5) didDrag = true;
        updateFromEvent(e);
        e.preventDefault();
    }

    function onPointerUp(e) {
        if (!dragging) return;
        dragging = false;
        if (e.target.releasePointerCapture) {
            try { e.target.releasePointerCapture(e.pointerId); } catch (_) {}
        }
        // If user did NOT drag (just clicked), open fullscreen preview
        if (!didDrag) {
            openComparisonPreview();
        }
    }

    function updateFromEvent(e) {
        var clientX = e.clientX;
        if (e.touches && e.touches.length > 0) clientX = e.touches[0].clientX;
        if (clientX == null) return;
        setSplit(pointerToPercent(clientX));
    }

    /** Open a fullscreen lightbox preview of the comparison base image */
    function openComparisonPreview() {
        var baseImg = document.querySelector('#img_comp_base img');
        if (!baseImg || !baseImg.src) return;
        // Reuse the existing lightbox modal from imageviewer.js if available
        var lb = document.getElementById('lightboxModal');
        var modalImg = document.getElementById('modalImage');
        if (lb && modalImg) {
            modalImg.src = baseImg.src;
            lb.style.display = 'flex';
            document.body.classList.add('lightbox-open');
            lb.focus();
        }
    }

    function bindDragEvents() {
        var row = document.getElementById('img_comp_row');
        var handle = document.getElementById('img_comp_handle');
        if (!row || !handle) return;
        if (handle.__sdCompBound) return;
        handle.__sdCompBound = true;

        // Use pointer events for unified mouse/touch support
        row.addEventListener('pointerdown', onPointerDown);
        row.addEventListener('pointermove', onPointerMove);
        row.addEventListener('pointerup', onPointerUp);
        row.addEventListener('pointercancel', onPointerUp);
        // Also allow dragging when clicking anywhere on the row
        row.style.cursor = 'ew-resize';
        // Double-click also opens preview explicitly
        row.addEventListener('dblclick', function(e) {
            openComparisonPreview();
            e.preventDefault();
        });
    }

    // ============================================================
    //  Public API (called from Python _js handlers)
    // ============================================================

    window.ImgCompLoader = {
        /** Load images from the two user upload components (pnginfo_image / pnginfo_image_b)
         *  into the comparison slider. These are the actual upload slots above the comparison panel. */
        loadImage: function(mode) {
            var aRoot = document.getElementById('pnginfo_image');
            var bRoot = document.getElementById('pnginfo_image_b');
            if (!aRoot || !bRoot) return;

            var srcA = extractImageSrc(aRoot);
            var srcB = extractImageSrc(bRoot);
            applyComparisonSources(srcA, srcB);

            // Gradio may render the upload preview before assigning its URL.
            // Retry briefly so the comparison does not depend on click timing.
            if ((!srcA || !srcB) && !window.__sdCompRetryTimer) {
                var attempts = 0;
                window.__sdCompRetryTimer = setInterval(function() {
                    attempts++;
                    var nextA = extractImageSrc(aRoot);
                    var nextB = extractImageSrc(bRoot);
                    if (nextA || nextB) applyComparisonSources(nextA, nextB);
                    if ((nextA && nextB) || attempts >= 20) {
                        clearInterval(window.__sdCompRetryTimer);
                        window.__sdCompRetryTimer = null;
                    }
                }, 150);
            }
        },

        /** Swap image A and B. */
        swapImage: function() {
            var baseImg = document.querySelector('#img_comp_base img');
            var topImg = document.querySelector('#img_comp_top img');
            if (!baseImg || !topImg) return;
            var tmp = baseImg.src;
            baseImg.src = topImg.src;
            topImg.src = tmp;
        }
    };

    window.ImageComparator = {
        /** Reset split position to center. */
        reset: function() {
            setSplit(50);
        },

        /** Initialize the comparison slider. Safe to call multiple times. */
        init: function() {
            if (buildComparisonDOM()) {
                bindDragEvents();
                if (!initialized) {
                    initialized = true;
                    setSplit(50);
                }
            }
        }
    };

    // ============================================================
    //  Auto-initialize after Gradio renders
    // ============================================================

    function tryInit() {
        if (buildComparisonDOM()) {
            bindDragEvents();
            setSplit(50);
            return true;
        }
        return false;
    }

    function waitForGradio() {
        var attempts = 0;
        var maxAttempts = 100; // ~20s
        (function tick() {
            attempts++;
            if (tryInit()) {
                initialized = true;
                // Observe #img_comp_row for late image src changes
                observeImageChanges();
                return;
            }
            if (attempts >= maxAttempts) return;
            setTimeout(tick, 200);
        })();
    }

    function observeImageChanges() {
        // Watch the actual user upload components, not the dummy comparison placeholders
        var aRoot = document.getElementById('pnginfo_image');
        var bRoot = document.getElementById('pnginfo_image_b');
        if (!aRoot || !bRoot || typeof MutationObserver === 'undefined') return;

        var observer = new MutationObserver(function() {
            // When Gradio swaps images, sync our comparison images
            window.ImgCompLoader.loadImage('auto');
        });

        [aRoot, bRoot].forEach(function(root) {
            observer.observe(root, {
                childList: true,
                subtree: true,
                attributes: true,
                attributeFilter: ['src', 'style', 'data-src', 'data-url', 'data-preview']
            });
        });

        // Some Gradio previews update canvas pixels without a useful mutation.
        [aRoot, bRoot].forEach(function(root) {
            root.addEventListener('load', function() {
                window.ImgCompLoader.loadImage('auto');
            }, true);
        });
    }

    // Start after DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', waitForGradio);
    } else {
        waitForGradio();
    }
})();
