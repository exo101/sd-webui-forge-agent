(function () {
    const TAB_PANEL_PATTERN = /_ad_tab(?:_\d+(?:st|nd|rd|th))?$/;
    const STORAGE_PREFIX = "adetailer.tabAlias.";

    function ordinal(n) {
        const d = {1: "st", 2: "nd", 3: "rd"};
        return `${n}${n % 100 >= 11 && n % 100 <= 13 ? "th" : d[n % 10] || "th"}`;
    }

    function fallbackLabel(root) {
        const match = root.id.match(/_ad_tab_alias(?:_(\d+)(?:st|nd|rd|th))?$/);
        if (!match || !match[1]) return "1st";
        return ordinal(Number(match[1]));
    }

    function textInput(root) {
        return root.querySelector("textarea, input");
    }

    function storageKey(root) {
        return STORAGE_PREFIX + root.id;
    }

    function readAlias(root) {
        try {
            return localStorage.getItem(storageKey(root));
        } catch {
            return null;
        }
    }

    function saveAlias(root, alias) {
        try {
            const key = storageKey(root);
            if (alias) {
                localStorage.setItem(key, alias);
            } else {
                localStorage.removeItem(key);
            }
        } catch {
            // Browser storage can be disabled; the visual rename still works.
        }
    }

    function tabButton(panel) {
        const tabs = panel.closest(".tabs");
        if (!tabs) return null;

        const tabNav = tabs.querySelector(".tab-nav, .tabnav");
        if (!tabNav) return null;

        const panels = Array.from(tabs.querySelectorAll("[id*='_adetailer_ad_tab']")).filter((item) =>
            TAB_PANEL_PATTERN.test(item.id)
        );
        const index = panels.indexOf(panel);
        if (index < 0) return null;

        return tabNav.querySelectorAll("button")[index] || null;
    }

    function bindAlias(root) {
        if (!root.id || !root.id.includes("_adetailer_ad_tab_alias")) return;

        const input = textInput(root);
        if (!input) return;

        const panelId = root.id.replace("_ad_tab_alias", "_ad_tab");
        const panel = document.getElementById(panelId);
        const button = panel ? tabButton(panel) : null;
        root.dataset.adetailerAliasFallback = root.dataset.adetailerAliasFallback || fallbackLabel(root);

        const saved = readAlias(root);
        if (saved !== null && input.value !== saved) {
            input.value = saved;
        }

        if (!input.dataset.adetailerAliasBound) {
            input.dataset.adetailerAliasBound = "true";
            input.addEventListener("input", updateAliasAndSave);
            input.addEventListener("change", updateAliasAndSave);
        }

        if (button) updateAlias({currentTarget: input});
    }

    function updateAlias(event) {
        const input = event.currentTarget;
        const root = input.closest("[id*='_adetailer_ad_tab_alias']");
        if (!root) return;

        const panelId = root.id.replace("_ad_tab_alias", "_ad_tab");
        const panel = document.getElementById(panelId);
        const button = panel ? tabButton(panel) : null;
        if (!button) return;

        const alias = input.value.trim();
        const label = alias || root.dataset.adetailerAliasFallback || fallbackLabel(root);
        if (button.textContent.trim() !== label) {
            button.textContent = label;
        }
    }

    function updateAliasAndSave(event) {
        const input = event.currentTarget;
        const root = input.closest("[id*='_adetailer_ad_tab_alias']");
        if (!root) return;

        saveAlias(root, input.value.trim());
        updateAlias(event);
    }

    function updateAll() {
        document.querySelectorAll("[id*='_adetailer_ad_tab_alias']").forEach((root) => {
            const input = textInput(root);
            if (input) updateAlias({currentTarget: input});
        });
    }

    function bindAll() {
        document.querySelectorAll("[id*='_adetailer_ad_tab_alias']").forEach(bindAlias);
        updateAll();
    }

    function start() {
        bindAll();
        document.addEventListener("click", () => setTimeout(updateAll, 0), true);

        if (typeof onAfterUiUpdate === "function") {
            onAfterUiUpdate(bindAll);
            return;
        }

        let attempts = 0;
        const timer = setInterval(() => {
            bindAll();
            attempts += 1;
            if (attempts >= 30) clearInterval(timer);
        }, 500);
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(start);
    } else if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start);
    } else {
        start();
    }
})();
