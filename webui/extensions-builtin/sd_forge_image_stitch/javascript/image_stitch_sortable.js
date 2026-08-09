// 等待 DOM 加载完成
window.addEventListener('DOMContentLoaded', function() {
    // 查找多图拼接参考的 Gallery 组件
    const galleryElement = document.getElementById('image_stitch_ref_latent');
    
    if (galleryElement) {
        // 找到 Gallery 组件，添加拖拽排序功能
        makeGallerySortable(galleryElement);
    } else {
        // 如果初始未找到，使用 MutationObserver 监听 DOM 变化
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                mutation.addedNodes.forEach(function(node) {
                    if (node.id === 'image_stitch_ref_latent') {
                        makeGallerySortable(node);
                        observer.disconnect();
                    }
                });
            });
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
});

// ===== Pose 素材库拖拽功能 =====

/**
 * 将插件中的提示词复制到主界面的正面提示词输入框
 */
function copyPromptToMainInterface() {
    try {
        console.log('[Image Stitch] 开始复制提示词到主界面...');
        
        // 获取插件内的提示词输入框 - 通过 label 文本来定位
        let pluginPromptInput = null;
        const allTextareas = document.querySelectorAll('textarea');
        
        // 遍历所有 textarea，找到"输入新提示词"标签对应的那个
        for (let textarea of allTextareas) {
            // 查找父元素中是否包含"输入新提示词"标签
            const parent = textarea.closest('.gradio-block');
            if (parent) {
                const label = parent.querySelector('label');
                if (label && label.textContent.includes('输入新提示词')) {
                    pluginPromptInput = textarea;
                    console.log('[Image Stitch] ✅ 找到插件提示词输入框');
                    break;
                }
            }
        }
        
        if (!pluginPromptInput) {
            console.warn('[Image Stitch] ❌ 未找到插件提示词输入框');
            showNotification('❌ 未找到插件输入框', 'error');
            return false;
        }
        
        const promptText = pluginPromptInput.value;
        
        if (!promptText || promptText.trim() === '') {
            console.warn('[Image Stitch] ⚠️ 提示词为空');
            showNotification('⚠️ 请先在输入框中输入或选择提示词', 'warning');
            return false;
        }
        
        console.log(`[Image Stitch] 📋 准备复制提示词: ${promptText.substring(0, 50)}...`);
        
        // 尝试找到文生图或图生图的正面提示词输入框
        let mainPromptInput = null;
        
        // 优先查找文生图提示词框
        mainPromptInput = document.getElementById('txt2img_prompt');
        if (mainPromptInput) {
            console.log('[Image Stitch] 🎯 找到文生图提示词框');
        }
        
        // 如果没找到，尝试图生图
        if (!mainPromptInput) {
            mainPromptInput = document.getElementById('img2img_prompt');
            if (mainPromptInput) {
                console.log('[Image Stitch] 🎯 找到图生图提示词框');
            }
        }
        
        if (mainPromptInput) {
            // 设置值
            mainPromptInput.value = promptText;
            
            // 触发 input 事件以通知 Gradio
            const inputEvent = new Event('input', { bubbles: true });
            mainPromptInput.dispatchEvent(inputEvent);
            
            // 触发 change 事件
            const changeEvent = new Event('change', { bubbles: true });
            mainPromptInput.dispatchEvent(changeEvent);
            
            // 额外触发 keyup 事件（某些组件可能需要）
            const keyupEvent = new KeyboardEvent('keyup', { bubbles: true });
            mainPromptInput.dispatchEvent(keyupEvent);
            
            console.log('[Image Stitch] ✅ 提示词已成功复制到主界面');
            
            // 显示成功提示
            showNotification('✅ 提示词已复制到主界面提示词框', 'success');
            return true;
        } else {
            console.warn('[Image Stitch] ❌ 未找到主界面提示词输入框');
            showNotification('❌ 未找到主界面提示词框，请确保在文生图或图生图页面', 'error');
            return false;
        }
    } catch (error) {
        console.error('[Image Stitch] ❌ 复制提示词失败:', error);
        showNotification('❌ 复制失败: ' + error.message, 'error');
        return false;
    }
}

/**
 * 显示通知消息
 */
function showNotification(message, type = 'info') {
    // 创建通知元素
    const notification = document.createElement('div');
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 24px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        z-index: 10000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        animation: slideIn 0.3s ease-out;
        ${type === 'success' ? 'background-color: #10b981;' : 
          type === 'error' ? 'background-color: #ef4444;' : 
          'background-color: #3b82f6;'}
    `;
    
    // 添加动画样式
    if (!document.getElementById('notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            @keyframes slideIn {
                from {
                    transform: translateX(400px);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            @keyframes slideOut {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(400px);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    document.body.appendChild(notification);
    
    // 3秒后自动移除
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

// 暴露函数到全局作用域
window.copyPromptToMainInterface = copyPromptToMainInterface;

console.log('[Image Stitch] 提示词传递功能已加载');

function makeGallerySortable(gallery) {
    console.log('[Image Stitch] 初始化拖拽排序功能');
    
    // 查找所有图片项
    const items = gallery.querySelectorAll('.gallery-item');
    
    if (items.length === 0) {
        // 如果没有图片项，等待新图片添加
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes.length > 0) {
                    const newItems = gallery.querySelectorAll('.gallery-item');
                    if (newItems.length > 0) {
                        console.log('[Image Stitch] 检测到新图片，重新初始化拖拽');
                        observer.disconnect();
                        initializeSortable(newItems);
                    }
                }
            });
        });
        
        observer.observe(gallery, {
            childList: true,
            subtree: true
        });
    } else {
        initializeSortable(items);
    }
}

function initializeSortable(items) {
    let draggedItem = null;
    
    items.forEach(item => {
        // 避免重复绑定
        if (item.hasAttribute('data-sortable-initialized')) {
            return;
        }
        item.setAttribute('data-sortable-initialized', 'true');
        
        // 添加拖拽事件监听器
        item.setAttribute('draggable', 'true');
        
        item.addEventListener('dragstart', function(e) {
            draggedItem = this;
            setTimeout(() => {
                this.style.opacity = '0.5';
                this.style.transform = 'scale(0.95)';
            }, 0);
            console.log('[Image Stitch] 开始拖拽');
        });
        
        item.addEventListener('dragend', function(e) {
            this.style.opacity = '1';
            this.style.transform = 'scale(1)';
            this.style.border = '';
            draggedItem = null;
            console.log('[Image Stitch] 拖拽结束');
        });
        
        item.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
        });
        
        item.addEventListener('dragenter', function(e) {
            e.preventDefault();
            this.style.border = '3px dashed #4CAF50';
            this.style.borderRadius = '8px';
        });
        
        item.addEventListener('dragleave', function(e) {
            this.style.border = '';
        });
        
        item.addEventListener('drop', function(e) {
            e.preventDefault();
            this.style.border = '';
            
            if (draggedItem !== this && draggedItem !== null) {
                const gallery = this.parentNode;
                const allItems = Array.from(gallery.children);
                const draggedIndex = allItems.indexOf(draggedItem);
                const dropIndex = allItems.indexOf(this);
                
                console.log(`[Image Stitch] 交换位置: ${draggedIndex} -> ${dropIndex}`);
                
                if (draggedIndex < dropIndex) {
                    gallery.insertBefore(draggedItem, this.nextSibling);
                } else {
                    gallery.insertBefore(draggedItem, this);
                }
                
                // 更新 Gallery 的值
                updateGalleryValue(gallery);
            }
        });
    });
    
    console.log(`[Image Stitch] 已为 ${items.length} 个图片项启用拖拽排序`);
}

function updateGalleryValue(gallery) {
    // 查找对应的 Gradio 组件
    const galleryId = gallery.id;
    if (galleryId) {
        // 触发 Gallery 的 change 事件，通知后端状态变化
        const event = new Event('change', { bubbles: true });
        gallery.dispatchEvent(event);
        
        // 也触发 input 事件以确保兼容性
        const inputEvent = new Event('input', { bubbles: true });
        gallery.dispatchEvent(inputEvent);
        
        console.log('[Image Stitch] Gallery 排序已更新并同步到后端');
    }
}
