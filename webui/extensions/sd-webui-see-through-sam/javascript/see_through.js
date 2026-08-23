// See-Through Plugin JavaScript

console.log('See-Through plugin loaded');

// 等待Gradio应用加载
function waitForGradioApp() {
    if (typeof gradioApp !== 'undefined') {
        initializeSeeThrough();
    } else {
        setTimeout(waitForGradioApp, 100);
    }
}

function initializeSeeThrough() {
    console.log('Initializing See-Through plugin');
    
    // 监听选项变化
    const seeThroughEnabled = gradioApp().querySelector('#see_through_enabled');
    if (seeThroughEnabled) {
        seeThroughEnabled.addEventListener('change', function() {
            console.log('See-Through enabled:', this.checked);
        });
    }
    
    // 添加自定义样式
    const style = document.createElement('style');
    style.textContent = `
        .see-through-container {
            border: 1px solid #444;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
            background: rgba(0, 0, 0, 0.2);
        }
        .see-through-button {
            margin: 4px;
        }
    `;
    document.head.appendChild(style);
}

// 初始化
waitForGradioApp();
