// ============================================================
// 教程中心插件 - 分类切换
// ============================================================

// 全局切换函数
function switchTutorialCategory(cat) {
    'use strict';
    var container = document.getElementById('tutorial-content-area');
    if (!container) return;

    var btns = container.querySelectorAll('.tutorial-cat-btn');
    for (var i = 0; i < btns.length; i++) {
        btns[i].classList.remove('active');
        if (btns[i].getAttribute('data-category') === cat) {
            btns[i].classList.add('active');
        }
    }

    var pages = container.querySelectorAll('.tutorial-category-page');
    for (var i = 0; i < pages.length; i++) {
        if (pages[i].getAttribute('data-category') === cat) {
            pages[i].style.display = '';
        } else {
            pages[i].style.display = 'none';
        }
    }
}

// 绑定按钮事件
function bindTutorialButtons() {
    var container = document.getElementById('tutorial-content-area');
    if (!container) return false;

    var btns = container.querySelectorAll('.tutorial-cat-btn');
    if (btns.length === 0) return false;

    for (var i = 0; i < btns.length; i++) {
        // 移除旧的监听器（防重复绑定）
        btns[i].removeEventListener('click', tutorialClickHandler);
        // 绑定新监听器
        btns[i].addEventListener('click', tutorialClickHandler);
    }
    return true;
}

function tutorialClickHandler() {
    var cat = this.getAttribute('data-category');
    if (cat) {
        switchTutorialCategory(cat);
    }
}

// 尝试绑定（带重试）
(function() {
    var attempts = 0;
    var maxAttempts = 20;

    function tryBind() {
        if (bindTutorialButtons()) {
            console.log('[教程中心] 初始化完成');
            return;
        }
        attempts++;
        if (attempts < maxAttempts) {
            setTimeout(tryBind, 500);
        }
    }

    // 页面加载完成后尝试
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', tryBind);
    } else {
        setTimeout(tryBind, 300);
    }

    // 监听 Gradio 重新渲染
    document.addEventListener('gradio:render', function() {
        setTimeout(tryBind, 300);
    });

    // 使用 MutationObserver 监听 DOM 变化
    var observer = new MutationObserver(function() {
        if (document.getElementById('tutorial-content-area')) {
            bindTutorialButtons();
        }
    });
    if (document.body) {
        observer.observe(document.body, { childList: true, subtree: true });
    } else {
        document.addEventListener('DOMContentLoaded', function() {
            observer.observe(document.body, { childList: true, subtree: true });
        });
    }
})();

console.log('[教程中心] 插件已加载');