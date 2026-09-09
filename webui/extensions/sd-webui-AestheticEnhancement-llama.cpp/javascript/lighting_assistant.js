/**
 * 打光辅助插件 - Lighting Assistant
 * 前端交互脚本
 */

(function() {
    'use strict';

    // 等待 DOM 加载完成
    document.addEventListener('DOMContentLoaded', function() {
        console.log('打光辅助插件: 初始化中...');
        initLightingAssistant();
    });

    // 也支持动态加载的情况
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        setTimeout(initLightingAssistant, 100);
    }

    function initLightingAssistant() {
        // 添加发送到提示词框的功能
        enhanceSendButtons();
        console.log('打光辅助插件: 初始化完成');
    }

    /**
     * 增强发送按钮功能
     */
    function enhanceSendButtons() {
        // 监听发送按钮点击
        document.addEventListener('click', function(e) {
            const target = e.target;
            
            // 检查是否是发送按钮
            if (target.matches('[data-send-target]')) {
                const targetId = target.dataset.sendTarget;
                const sourceId = target.dataset.source;
                
                sendToPrompt(sourceId, targetId);
            }
        });
    }

    /**
     * 发送关键词到提示词框
     * @param {string} sourceId - 源文本框ID
     * @param {string} targetId - 目标提示词框ID
     */
    function sendToPrompt(sourceId, targetId) {
        const sourceElement = document.getElementById(sourceId);
        const targetElement = document.querySelector(targetId + ' textarea') || document.getElementById(targetId);
        
        if (!sourceElement || !targetElement) {
            console.warn('打光辅助插件: 找不到源或目标元素');
            return;
        }

        const keywords = sourceElement.value || sourceElement.textContent;
        const currentPrompt = targetElement.value || '';

        // 追加关键词
        if (currentPrompt.trim()) {
            targetElement.value = currentPrompt.trim() + ', ' + keywords;
        } else {
            targetElement.value = keywords;
        }

        // 触发 input 事件以更新 Gradio 状态
        targetElement.dispatchEvent(new Event('input', { bubbles: true }));
        targetElement.dispatchEvent(new Event('change', { bubbles: true }));

        // 显示成功提示
        showToast('关键词已发送到提示词框');
    }

    /**
     * 显示提示消息
     * @param {string} message - 消息内容
     */
    function showToast(message) {
        // 检查是否已存在 toast
        let toast = document.getElementById('lighting-assistant-toast');
        
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'lighting-assistant-toast';
            toast.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 12px 24px;
                border-radius: 8px;
                z-index: 10000;
                font-size: 14px;
                transition: opacity 0.3s;
            `;
            document.body.appendChild(toast);
        }

        toast.textContent = message;
        toast.style.opacity = '1';

        setTimeout(function() {
            toast.style.opacity = '0';
            setTimeout(function() {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, 2000);
    }

    /**
     * 快速预设选择
     */
    const PRESETS = {
        'portrait-dramatic': {
            directions: ['伦勃朗光'],
            qualities: ['硬光'],
            colors: ['暖光'],
            effects: ['电影光'],
            contrasts: ['明暗对照'],
            atmospheres: ['戏剧性']
        },
        'portrait-soft': {
            directions: ['环形光'],
            qualities: ['柔光'],
            colors: ['中性光'],
            effects: ['眼神光'],
            contrasts: ['平衡对比'],
            atmospheres: ['柔和']
        },
        'cinematic-movie': {
            directions: ['侧面光'],
            qualities: ['硬光'],
            colors: ['黄金时刻'],
            effects: ['电影光', '体积光'],
            contrasts: ['明暗对照'],
            atmospheres: ['戏剧性']
        },
        'cyberpunk': {
            directions: ['侧逆光'],
            qualities: ['硬光'],
            colors: ['霓虹光'],
            effects: ['轮廓光', '光晕'],
            contrasts: ['低对比'],
            atmospheres: ['科幻']
        },
        'dreamy-fantasy': {
            directions: ['正逆光'],
            qualities: ['柔光'],
            colors: ['黄金时刻'],
            effects: ['丁达尔光', '光晕'],
            contrasts: ['高对比'],
            atmospheres: ['梦幻']
        },
        'horror': {
            directions: ['底光'],
            qualities: ['硬光'],
            colors: ['冷光'],
            effects: ['体积光'],
            contrasts: ['低对比'],
            atmospheres: ['恐怖']
        }
    };

    /**
     * 应用预设
     * @param {string} presetName - 预设名称
     */
    window.applyLightingPreset = function(presetName) {
        const preset = PRESETS[presetName];
        if (!preset) {
            console.warn('打光辅助插件: 未找到预设', presetName);
            return;
        }

        // 这里需要与 Gradio 组件交互
        // 由于 Gradio 的动态特性，这个功能需要在 Python 端实现
        console.log('应用预设:', presetName, preset);
    };

    // 导出函数供外部调用
    window.LightingAssistant = {
        sendToPrompt: sendToPrompt,
        showToast: showToast,
        PRESETS: PRESETS
    };

})();
