// OpenPose Editor integration for Forge Neo's ControlNet.
// Adds an "Edit" button to each ControlNet unit's generated (openpose) preview.
// Clicking it opens the locally mounted Vue editor at /openpose_editor_index
// inside a modal iframe. Communication with the editor uses the postMessage
// protocol implemented in the editor's App.vue:
//   - editor -> parent: { ready: true }
//   - parent -> editor: { modalId, imageURL?, poseURL? }
//   - editor -> parent: { modalId, poseURL, poses }
(function () {
    const EDITOR_URL = '/openpose_editor_index';
    const AllCnetTabs = new Set();

    function gradioApp() {
        if (window.gradioApp) return window.gradioApp();
        const elems = document.getElementsByTagName('gradio-app');
        const sr = elems.length ? elems[0].shadowRoot : null;
        return sr ? sr : document;
    }

    let activeModal = null;

    function closeModal() {
        if (activeModal && activeModal.overlay && activeModal.overlay.parentNode) {
            activeModal.overlay.parentNode.removeChild(activeModal.overlay);
        }
        activeModal = null;
    }

    function openEditor(tabContext) {
        const { updatePreviewPose, poseURL, imageURL } = tabContext;

        const overlay = document.createElement('div');
        overlay.style.cssText =
            'position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:99999;' +
            'display:flex;align-items:center;justify-content:center;';

        const modal = document.createElement('div');
        modal.style.cssText =
            'position:relative;width:92vw;height:92vh;background:#fff;' +
            'border-radius:8px;overflow:hidden;box-shadow:0 8px 40px rgba(0,0,0,.4);';

        const closeBtn = document.createElement('button');
        closeBtn.textContent = '✕';
        closeBtn.title = 'Close';
        closeBtn.style.cssText =
            'position:absolute;top:6px;right:8px;z-index:20;width:32px;height:32px;' +
            'border:0;border-radius:50%;background:rgba(0,0,0,.5);color:#fff;' +
            'font-size:16px;line-height:1;cursor:pointer;';

        const iframe = document.createElement('iframe');
        iframe.src = EDITOR_URL;
        iframe.style.cssText = 'width:100%;height:100%;border:0;display:block;';

        modal.appendChild(closeBtn);
        modal.appendChild(iframe);
        overlay.appendChild(modal);
        document.body.appendChild(overlay);

        activeModal = {
            overlay: overlay,
            iframe: iframe,
            modalId: 'openpose_editor_' + Date.now(),
            poseURL: poseURL,
            imageURL: imageURL,
            updatePreviewPose: updatePreviewPose,
        };

        closeBtn.addEventListener('click', closeModal);
        overlay.addEventListener('click', function (e) {
            if (e.target === overlay) closeModal();
        });
    }

    function setupTab(tab) {
        if (AllCnetTabs.has(tab)) return;
        AllCnetTabs.add(tab);

        const generatedImageGroup = tab.querySelector('.cnet-generated-image-group');
        if (!generatedImageGroup) return;

        const allowPreviewCheckbox = tab.querySelector('.cnet-allow-preview input');
        const downloadLink = generatedImageGroup.querySelector('.cnet-download-pose a');
        const poseTextbox = generatedImageGroup.querySelector('.cnet-pose-json textarea');
        const renderButton = generatedImageGroup.querySelector('.cnet-render-pose');
        const controlGroup = generatedImageGroup.querySelector('.cnet-generated-image-control-group');
        if (!controlGroup || !poseTextbox || !renderButton) return;

        function updatePreviewPose(poseURL) {
            if (allowPreviewCheckbox && !allowPreviewCheckbox.checked) {
                allowPreviewCheckbox.click();
            }
            // Re-query: the download link element may have been replaced.
            const liveDownloadLink =
                generatedImageGroup.querySelector('.cnet-download-pose a');
            if (liveDownloadLink != null) liveDownloadLink.href = poseURL;
            poseTextbox.value = poseURL;
            if (typeof updateInput === 'function') updateInput(poseTextbox);
            renderButton.click();
        }

        const editBtn = document.createElement('a');
        editBtn.textContent = 'Edit';
        editBtn.title = 'Edit pose in OpenPose Editor';
        editBtn.className = 'cnet-edit-pose';
        editBtn.style.cssText = 'cursor:pointer;margin:0 4px;';

        editBtn.addEventListener('click', function () {
            // Re-query the download link on every click. The ControlNet
            // `gr.HTML` element is replaced (innerHTML swapped) whenever a new
            // pose is generated, so a reference captured at setup time would
            // point at a stale, empty <a>.
            let poseURL = '';
            const liveDownloadLink =
                generatedImageGroup.querySelector('.cnet-download-pose a');
            if (
                liveDownloadLink &&
                liveDownloadLink.href &&
                liveDownloadLink.href.indexOf('data:') === 0
            ) {
                poseURL = liveDownloadLink.href;
            }

            // Background image (the generated preview), if any.
            let imageURL = '';
            const img = generatedImageGroup.querySelector('img.forge-image');
            if (img && img.src) imageURL = img.src;

            // Open the editor even without a generated pose: the editor starts
            // with a default person so the user can create a pose from scratch.
            openEditor({
                updatePreviewPose: updatePreviewPose,
                poseURL: poseURL,
                imageURL: imageURL,
            });
        });

        controlGroup.appendChild(editBtn);
    }

    function setupAll() {
        const tabs = gradioApp().querySelectorAll('#controlnet .tabitem');
        tabs.forEach(setupTab);
    }

    window.addEventListener('message', function (event) {
        const msg = event.data;
        if (!msg) return;

        if (msg.ready === true) {
            if (activeModal) {
                activeModal.iframe.contentWindow.postMessage({
                    modalId: activeModal.modalId,
                    imageURL: activeModal.imageURL,
                    poseURL: activeModal.poseURL,
                }, '*');
            }
            return;
        }

        if (
            msg.modalId &&
            activeModal &&
            msg.modalId === activeModal.modalId &&
            msg.poseURL
        ) {
            activeModal.updatePreviewPose(msg.poseURL);
            closeModal();
        }
    });

    if (window.onUiLoaded) {
        onUiLoaded(setupAll);
    } else {
        document.addEventListener('DOMContentLoaded', setupAll);
    }

    // ControlNet units can be created lazily; re-scan periodically.
    setInterval(setupAll, 2000);
})();
