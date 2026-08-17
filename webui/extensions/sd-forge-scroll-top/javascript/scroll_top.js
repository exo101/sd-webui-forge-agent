(() => {
    "use strict";

    const EXT_ID = "forge-scroll-top-ball";
    const STORAGE_KEY = "forge-scroll-top-ball-position-v1";
    const DRAG_THRESHOLD = 5;

    let ball = null;
    let dragging = false;
    let moved = false;
    let pointerId = null;
    let startX = 0;
    let startY = 0;
    let startLeft = 0;
    let startTop = 0;

    function appRoot() {
        try {
            if (typeof gradioApp === "function") return gradioApp();
        } catch (_) {}
        return document;
    }

    function clamp(v, min, max) {
        return Math.max(min, Math.min(max, v));
    }

    function savePosition() {
        if (!ball) return;
        const rect = ball.getBoundingClientRect();
        localStorage.setItem(STORAGE_KEY, JSON.stringify({
            left: rect.left,
            top: rect.top
        }));
    }

    function restorePosition() {
        if (!ball) return;
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (!raw) return;
            const p = JSON.parse(raw);
            if (!Number.isFinite(p.left) || !Number.isFinite(p.top)) return;

            const rect = ball.getBoundingClientRect();
            const maxLeft = Math.max(8, window.innerWidth - rect.width - 8);
            const maxTop = Math.max(8, window.innerHeight - rect.height - 8);

            ball.style.left = clamp(p.left, 8, maxLeft) + "px";
            ball.style.top = clamp(p.top, 8, maxTop) + "px";
            ball.style.right = "auto";
            ball.style.bottom = "auto";
        } catch (_) {}
    }

    function keepOnScreen() {
        if (!ball) return;
        const rect = ball.getBoundingClientRect();
        if (rect.left < 8 || rect.top < 8 ||
            rect.right > window.innerWidth - 8 ||
            rect.bottom > window.innerHeight - 8) {
            ball.style.left = clamp(rect.left, 8, Math.max(8, window.innerWidth - rect.width - 8)) + "px";
            ball.style.top = clamp(rect.top, 8, Math.max(8, window.innerHeight - rect.height - 8)) + "px";
            ball.style.right = "auto";
            ball.style.bottom = "auto";
            savePosition();
        }
    }

    function isScrollable(el) {
        if (!el || el === document || el === document.body || el === document.documentElement) return false;
        const style = getComputedStyle(el);
        const oy = style.overflowY;
        return (oy === "auto" || oy === "scroll" || oy === "overlay") &&
               el.scrollHeight > el.clientHeight + 4;
    }

    function collectScrollTargets() {
        const targets = [];
        const seen = new Set();

        const add = (el) => {
            if (!el || seen.has(el)) return;
            seen.add(el);
            targets.push(el);
        };

        add(document.scrollingElement);
        add(document.documentElement);
        add(document.body);

        const root = appRoot();
        if (root && root.querySelectorAll) {
            // Forge / Gradio may place the actual page in an internal scrolling container.
            root.querySelectorAll("*").forEach((el) => {
                try {
                    if (isScrollable(el) && el.scrollTop > 0) add(el);
                } catch (_) {}
            });
        }

        return targets;
    }

    function scrollEverythingToTop() {
        const targets = collectScrollTargets();
        let didScroll = false;

        for (const el of targets) {
            try {
                if (el && typeof el.scrollTo === "function" && el.scrollTop > 0) {
                    el.scrollTo({ top: 0, behavior: "smooth" });
                    didScroll = true;
                }
            } catch (_) {}
        }

        try {
            if (window.scrollY > 0 || !didScroll) {
                window.scrollTo({ top: 0, behavior: "smooth" });
            }
        } catch (_) {
            window.scrollTo(0, 0);
        }
    }

    function createBall() {
        if (document.getElementById(EXT_ID)) {
            ball = document.getElementById(EXT_ID);
            return;
        }

        ball = document.createElement("button");
        ball.id = EXT_ID;
        ball.type = "button";
        ball.setAttribute("aria-label", "回到顶部");
        ball.title = "回到顶部（可拖动）";
        ball.innerHTML = `
            <span class="forge-scroll-top-arrow" aria-hidden="true">↑</span>
            <span class="forge-scroll-top-tip">回顶</span>
        `;

        document.body.appendChild(ball);
        restorePosition();

        ball.addEventListener("click", (e) => {
            if (moved) {
                e.preventDefault();
                e.stopPropagation();
                moved = false;
                return;
            }
            scrollEverythingToTop();
        });

        ball.addEventListener("pointerdown", (e) => {
            if (e.button !== undefined && e.button !== 0) return;

            pointerId = e.pointerId;
            dragging = true;
            moved = false;

            const rect = ball.getBoundingClientRect();
            startX = e.clientX;
            startY = e.clientY;
            startLeft = rect.left;
            startTop = rect.top;

            ball.setPointerCapture?.(pointerId);
            ball.classList.add("dragging");
            e.preventDefault();
        });

        ball.addEventListener("pointermove", (e) => {
            if (!dragging || (pointerId !== null && e.pointerId !== pointerId)) return;

            const dx = e.clientX - startX;
            const dy = e.clientY - startY;

            if (!moved && Math.hypot(dx, dy) >= DRAG_THRESHOLD) moved = true;
            if (!moved) return;

            const rect = ball.getBoundingClientRect();
            const maxLeft = Math.max(8, window.innerWidth - rect.width - 8);
            const maxTop = Math.max(8, window.innerHeight - rect.height - 8);

            ball.style.left = clamp(startLeft + dx, 8, maxLeft) + "px";
            ball.style.top = clamp(startTop + dy, 8, maxTop) + "px";
            ball.style.right = "auto";
            ball.style.bottom = "auto";

            e.preventDefault();
        });

        const endDrag = (e) => {
            if (!dragging) return;
            if (pointerId !== null && e.pointerId !== undefined && e.pointerId !== pointerId) return;

            dragging = false;
            ball.classList.remove("dragging");
            try { ball.releasePointerCapture?.(pointerId); } catch (_) {}
            pointerId = null;

            if (moved) savePosition();

            // Prevent the pointerup-generated click from being interpreted as "back to top".
            setTimeout(() => { moved = false; }, 80);
        };

        ball.addEventListener("pointerup", endDrag);
        ball.addEventListener("pointercancel", endDrag);

        // Right click resets to the default bottom-right position.
        ball.addEventListener("contextmenu", (e) => {
            e.preventDefault();
            localStorage.removeItem(STORAGE_KEY);
            ball.style.left = "";
            ball.style.top = "";
            ball.style.right = "";
            ball.style.bottom = "";
            ball.classList.add("forge-scroll-top-reset");
            setTimeout(() => ball?.classList.remove("forge-scroll-top-reset"), 220);
        });

        window.addEventListener("resize", keepOnScreen, { passive: true });
    }

    function init() {
        if (document.body) createBall();
    }

    // A1111 / Forge / Forge Neo compatible initialization paths.
    if (typeof onUiLoaded === "function") {
        onUiLoaded(init);
    } else if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init, { once: true });
    } else {
        init();
    }

    // Some Forge Neo UI rebuilds can replace body-level extension nodes.
    if (typeof onUiUpdate === "function") {
        onUiUpdate(() => {
            if (!document.getElementById(EXT_ID)) createBall();
        });
    }
})();