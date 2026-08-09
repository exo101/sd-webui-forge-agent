// Selects a model by pressing on card
function select_model(model_name, event, bool = false, content_type = null, sendToBrowser = false) {
    if (event) {
        var className = event.target.className;
        if (className.includes('custom-checkbox') || className.includes('model-checkbox')) {
            return;
        }
    }

    let output;
    if (sendToBrowser) {
        output = gradioApp().querySelector('#send_to_browser textarea');
    } else {
        output = bool ? gradioApp().querySelector('#model_sent textarea') : gradioApp().querySelector('#model_select textarea');
    }

    if (output && model_name) {
        const randomNumber = Math.floor(Math.random() * 1000);
        const paddedNumber = String(randomNumber).padStart(3, '0');
        output.value = model_name + '.' + paddedNumber;
        updateInput(output);
    }

    if (content_type) {
        const outputType = gradioApp().querySelector('#type_sent textarea');
        const randomNumber = Math.floor(Math.random() * 1000);
        const paddedNumber = String(randomNumber).padStart(3, '0');
        outputType.value = content_type + '.' + paddedNumber;
        updateInput(outputType);
    }
}

// Clicks the first item in the browser cards list
function clickFirstFigureInColumn() {
    setTimeout(() => {
        const columnDiv = document.querySelector('.column.civmodellist');
        if (columnDiv) {
            const firstFigure = columnDiv.querySelector('figure');
            if (firstFigure) {
                firstFigure.click();
            }
        }
    }, 500);
}

// === ANXETY EDITs ===
// Changes the card size
function updateCardSize(width, height) {
    var styleSheet = document.styleSheets[0];
    var dimensionsKeyframes = `width: ${width}em !important; height: ${height}em !important;`;

    var fontSize = (width / 12) * 100;
    var textKeyframes = `font-size: ${fontSize}% !important;`;

    addOrUpdateRule(styleSheet, '.civmodelcard img', dimensionsKeyframes);
    addOrUpdateRule(styleSheet, '.civmodelcard .video-bg', dimensionsKeyframes);
    addOrUpdateRule(styleSheet, '.civmodelcard figcaption', textKeyframes);

    // Hide badges when tile size is less than 11
    var badgeDisplay = width < 11 ? 'none !important' : 'flex !important';
    addOrUpdateRule(styleSheet, '.model-type-badge', `display: ${badgeDisplay}`);
    addOrUpdateRule(styleSheet, '.nsfw-badge', `display: ${badgeDisplay}`);

    // Hide outdated-card delete button when tile size is less than 11 (same threshold as badges)
    var outdatedDeleteDisplay = width < 11 ? 'none !important' : 'flex !important';
    addOrUpdateRule(styleSheet, '.outdated-card-actions .delete-model-btn', `display: ${outdatedDeleteDisplay}`);
}

// Updates site with css insertions
function addOrUpdateRule(styleSheet, selector, newRules) {
    for (let i = 0; i < styleSheet.cssRules.length; i++) {
        let rule = styleSheet.cssRules[i];
        if (rule.selectorText === selector) {
            rule.style.cssText = newRules;
            return;
        }
    }
    styleSheet.insertRule(`${selector} { ${newRules} }`, styleSheet.cssRules.length);
}

// === ANXETY EDITs ===
// Updates card border
const cardSyncRetryState = {};
const pendingCardUpdates = new Set();

function applyPendingCardUpdates() {
    if (pendingCardUpdates.size === 0) return;
    const toApply = Array.from(pendingCardUpdates);
    pendingCardUpdates.clear();
    toApply.forEach((modelNameWithSuffix) => {
        updateCard(modelNameWithSuffix, false);
    });
}

// Watch for card containers appearing in the DOM (e.g. when user switches back to Browser/Update tab)
(function initCardUpdateObserver() {
    const observer = new MutationObserver((mutations) => {
        let foundContainer = false;
        for (const mutation of mutations) {
            for (const node of mutation.addedNodes) {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    if (
                        node.matches && (node.matches('.civmodellist') || node.matches('.civmodelcards'))
                    ) {
                        foundContainer = true;
                    } else if (node.querySelector) {
                        if (
                            node.querySelector('.civmodellist') ||
                            node.querySelector('.civmodelcards')
                        ) {
                            foundContainer = true;
                        }
                    }
                }
            }
        }
        if (foundContainer) {
            // Re-apply DOM filters after Gradio finishes injecting the new HTML
            setTimeout(reapplyFilters, 250);

            if (pendingCardUpdates.size > 0) {
                console.log('[updateCard] container appeared — applying', pendingCardUpdates.size, 'pending update(s)');
                // Small delay to let Gradio finish rendering cards inside the container
                setTimeout(applyPendingCardUpdates, 200);
            }
        }
    });
    observer.observe(document.documentElement, { childList: true, subtree: true });
})();

// Polling fallback: Gradio 4.x often injects cards via innerHTML which doesn't reliably
// trigger childList mutations for individual cards. We poll every 600ms to check if
// any visible container now has cards, and apply pending updates if so.
(function initCardUpdatePoller() {
    setInterval(() => {
        if (pendingCardUpdates.size === 0) return;

        const containers = [
            document.querySelector('.civmodellist'),
            document.querySelector('.civmodelcards')
        ].filter(Boolean);

        if (containers.length === 0) {
            console.log('[updateCard] poller: no containers in DOM');
            return;
        }

        // If any container has cards, apply pending updates directly.
        const hasCards = containers.some((container) => {
            return container.querySelectorAll('.civmodelcard').length > 0;
        });

        if (hasCards) {
            console.log('[updateCard] poller: cards detected — applying', pendingCardUpdates.size, 'pending update(s)');
            applyPendingCardUpdates();
            reapplyFilters();
        } else {
            // Containers exist but have no cards (Gradio cleared them on tab switch).
            // Force a refresh to reload cards with updated status.
            console.log('[updateCard] poller: containers empty — forcing refresh');
            pendingCardUpdates.clear();
            pressRefresh();
        }
    }, 600);
})();

function updateCard(modelNameWithSuffix, allowRefresh = true) {
    if (!modelNameWithSuffix || typeof modelNameWithSuffix !== 'string') {
        return;
    }

    const lastDotIndex = modelNameWithSuffix.lastIndexOf('.');
    if (lastDotIndex <= 0) {
        return;
    }

    const modelName = modelNameWithSuffix.slice(0, lastDotIndex);
    const suffix = modelNameWithSuffix.slice(lastDotIndex + 1);

    let additionalClassName = '';
    switch (suffix) {
        case 'None':
            additionalClassName = '';
            break;
        case 'Old':
            additionalClassName = 'civmodelcardoutdated';
            break;
        case 'New':
            additionalClassName = 'civmodelcardinstalled';
            break;
        default:
            return;
    }

    // Extract modelId from anywhere in the string (e.g. "Name (12345).New")
    // instead of requiring it at the end — the suffix (.New/.Old/.None) comes after.
    const modelIdMatch = modelNameWithSuffix.match(/\((-?\d+)\)/);
    const modelId = modelIdMatch ? modelIdMatch[1] : null;
    const statusClasses = ['civmodelcardinstalled', 'civmodelcardoutdated', 'civmodelcardcrossfamily'];

    // Search both Browser-mode (.civmodellist) and Update-mode (.civmodelcards) containers
    const containers = [
        document.querySelector('.civmodellist'),
        document.querySelector('.civmodelcards')
    ].filter(Boolean);

    console.log('[updateCard] called:', modelNameWithSuffix, 'modelId:', modelId, 'containers:', containers.length, 'allowRefresh:', allowRefresh);

    if (containers.length === 0) {
        // No card containers visible — user is on a different tab. Queue for later.
        console.log('[updateCard] no containers — queued for later');
        pendingCardUpdates.add(modelNameWithSuffix);
        return;
    }

    let matchedCount = 0;
    containers.forEach((parentDiv, idx) => {
        const cards = parentDiv.querySelectorAll('.civmodelcard');
        console.log('[updateCard] container', idx, 'has', cards.length, 'cards');
        cards.forEach((card) => {
            let cardMatches = false;
            let matchMethod = '';

            if (modelId) {
                const cardModelId = card.getAttribute('data-model-id');
                cardMatches = cardModelId === modelId;
                if (cardMatches) matchMethod = 'data-model-id';
            }

            // Backward compatibility for cards rendered before data-model-id existed.
            if (!cardMatches) {
                const onclickAttr = card.getAttribute('onclick');
                cardMatches = !!(onclickAttr && onclickAttr.includes(`select_model('${modelName}', event)`));
                if (cardMatches) matchMethod = 'onclick';
            }

            if (!cardMatches) {
                return;
            }

            matchedCount += 1;
            console.log('[updateCard] MATCH via', matchMethod, '— adding', additionalClassName);
            statusClasses.forEach((statusClass) => card.classList.remove(statusClass));
            if (additionalClassName) {
                card.classList.add(additionalClassName);
            }
        });
    });

    // Fallback: if the card is not yet in the DOM snapshot, retry briefly.
    // If retries exhaust, queue for when the user returns to the tab (via MutationObserver).
    // NOTE: allowRefresh=false skips pressRefresh() — used for post-download triggers
    // to avoid expensive API re-fetch (100 items) when a single card is missing.
    if (matchedCount === 0) {
        const retryCount = cardSyncRetryState[modelNameWithSuffix] || 0;
        console.log('[updateCard] no match — retry', retryCount + 1, '/ 4');
        if (retryCount < 4) {
            cardSyncRetryState[modelNameWithSuffix] = retryCount + 1;
            setTimeout(() => updateCard(modelNameWithSuffix, allowRefresh), 120);
        } else {
            delete cardSyncRetryState[modelNameWithSuffix];
            console.log('[updateCard] exhausted retries — queued for tab-switch');
            // Instead of forcing a full refresh, queue for later tab-switch
            pendingCardUpdates.add(modelNameWithSuffix);
            if (allowRefresh) {
                pressRefresh();
            }
        }
    } else {
        delete cardSyncRetryState[modelNameWithSuffix];
        pendingCardUpdates.delete(modelNameWithSuffix);
        console.log('[updateCard] success — updated', matchedCount, 'card(s)');
    }

    const hideInstalledToggle =
        gradioApp().querySelector('#toggle5 input[type="checkbox"]') ||
        gradioApp().querySelector('#toggle5L input[type="checkbox"]');
    if (hideInstalledToggle) {
        hideInstalled(hideInstalledToggle.checked);
    }
}

// === VIDEO HOVER-TO-PLAY ===
// Attach hover-to-play listeners to a single card element
function attachVideoHoverPlay(card) {
    const video = card.querySelector('video.video-bg');
    if (!video || video._hoverPlayAttached) return;
    video._hoverPlayAttached = true;

    card.addEventListener('mouseenter', () => {
        video.play().catch(() => {});
    });
    card.addEventListener('mouseleave', () => {
        video.pause();
        video.currentTime = 0;
    });
}

// Observe the model list container for newly injected cards
(function initVideoHoverObserver() {
    function attachAll(root) {
        root.querySelectorAll('.civmodelcard').forEach(attachVideoHoverPlay);
    }

    const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            for (const node of mutation.addedNodes) {
                if (node.nodeType !== 1) continue;
                if (node.classList && node.classList.contains('civmodelcard')) {
                    attachVideoHoverPlay(node);
                } else {
                    attachAll(node);
                }
            }
        }
    });

    function startObserver() {
        const container = document.querySelector('.civmodellist') || document.body;
        observer.observe(container, { childList: true, subtree: true });
        attachAll(container);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', startObserver);
    } else {
        startObserver();
    }
})();

// Enables refresh with alt+enter and ctrl+enter
function keydownHandler(e) {
    var handled = false;

    if (e.key !== undefined) {
        if (e.key == 'Enter' && (e.metaKey || e.ctrlKey || e.altKey)) handled = true;
    } else if (e.keyCode !== undefined) {
        if (e.keyCode == 13 && (e.metaKey || e.ctrlKey || e.altKey)) handled = true;
    }

    if (handled) {
        var currentTabContent = get_uiCurrentTabContent();
        if (currentTabContent && currentTabContent.id === 'tab_civitai_interface') {
            var refreshButton = currentTabContent.querySelector('#refreshBtn');
            if (!refreshButton) {
                refreshButton = currentTabContent.querySelector('#refreshBtnL');
            }
            if (refreshButton) {
                refreshButton.click();
            }

            e.preventDefault();
        }
    }
}
document.addEventListener('keydown', keydownHandler);

// Function to adjust alignment of Filter Accordion
function adjustFilterBoxAndButtons() {
    const element = document.querySelector('#filterBox') || document.querySelector('#filterBoxL');
    if (!element) return;

    const childDiv = element.querySelector('div:nth-child(3)');
    if (!childDiv) return;

    const isLargeScreen = window.innerWidth >= 1250;
    const isMediumScreen = window.innerWidth < 1250 && window.innerWidth > 915;
    const isNarrowScreen = window.innerWidth < 800;
    const modelBlocks = document.querySelectorAll('#civitai_preview_html .model-block');
    const civitInfo = document.querySelector('.civitai-version-info');

    if (modelBlocks) {
        modelBlocks.forEach((modelBlock) => {
            if (isNarrowScreen) {
                modelBlock.style.flexWrap = 'wrap';
                modelBlock.style.justifyContent = 'center';
            } else {
                modelBlock.style.flexWrap = 'nowrap';
                modelBlock.style.justifyContent = 'flex-start';
            }
        });
    }
    if (civitInfo) {
        if (window.innerWidth < 900) {
            civitInfo.style.flexWrap = 'wrap';
        } else {
            civitInfo.style.flexWrap = 'nowrap';
        }
    }

    childDiv.style.marginLeft = isLargeScreen ? '0px' : isMediumScreen ? `${1250 - window.innerWidth}px` : '0px';
    element.style.justifyContent = isLargeScreen || isMediumScreen ? 'center' : 'flex-start';

    const pageBtn1 = document.querySelector('#pageBtn1');
    const pageBtn2 = document.querySelector('#pageBtn2');
    const pageBox = document.querySelector('#pageBox');
    const pageBoxMobile = document.querySelector('#pageBoxMobile');

    if (window.innerWidth < 530) {
        childDiv.style.width = '300px';
        if (pageBoxMobile) {
            pageBtn1 && pageBoxMobile.appendChild(pageBtn1);
            pageBtn2 && pageBoxMobile.appendChild(pageBtn2);
            pageBoxMobile.style.paddingBottom = '15px';
        }
    } else {
        childDiv.style.width = '400px';
        if (pageBox) {
            pageBtn1 && pageBox.insertBefore(pageBtn1, pageBox.firstChild);
            pageBtn2 && pageBox.appendChild(pageBtn2);
            pageBoxMobile.style.paddingBottom = '0px';
        }
    }
}

// Calls the function above whenever the window is resized
window.addEventListener('resize', adjustFilterBoxAndButtons);

// Function to trigger refresh button with extra checks for page slider
function pressRefresh() {
    setTimeout(() => {
        const input = document.querySelector('#pageSlider > div:nth-child(2) > div > input');
        if (document.activeElement === input) {
            function keydownHandler(event) {
                if (event.key === 'Enter' || event.keyCode === 13) {
                    input.blur();
                    input.removeEventListener('keydown', keydownHandler);
                    input.removeEventListener('blur', blurHandler);
                }
            }

            function blurHandler() {
                input.removeEventListener('keydown', keydownHandler);
                input.removeEventListener('blur', blurHandler);
            }

            input.addEventListener('keydown', keydownHandler);
            input.addEventListener('blur', blurHandler);

            return;
        }
        let output = gradioApp().querySelector('#page_slider_trigger textarea');
        if (output) {
            const randomNumber = Math.floor(Math.random() * 1000);
            const paddedNumber = String(randomNumber).padStart(3, '0');
            output.value = paddedNumber;
            updateInput(output);
        }
    }, 200);
}

// Update SVG Icons based on dark theme or light theme
function updateSVGIcons() {
    const isDark = document.body.classList.contains('dark');
    const filterIconUrl = isDark
        ? 'https://gistcdn.githack.com/BlafKing/a20124cedafad23d4eecc1367ec22896/raw/04a4dae0771353377747dadf57c91d55bf841bed/filter-light.svg'
        : 'https://gistcdn.githack.com/BlafKing/686c3438f5d0d13e7e47135f25445ef3/raw/46477777faac7209d001829a171462d9a2ff1467/filter-dark.svg';
    const searchIconUrl = isDark
        ? 'https://gistcdn.githack.com/BlafKing/3f95619089bac3b4fd5470a986e1b3bb/raw/ebaa9cceee3436711eb560a7a65e151f1d651c6a/search-light.svg'
        : 'https://gistcdn.githack.com/BlafKing/57573592d5857e102a4bfde852f62639/raw/aa213e9e82d705651603507e26545eb0ffe60c90/search-dark.svg';

    if (isDark) {
    }

    const element = document.querySelector('#filterBox, #filterBoxL');
    const childDiv = element?.querySelector('div:nth-child(3)');

    if (childDiv) {
        childDiv.style.cssText = `box-shadow: ${isDark ? '#ffffff' : '#000000'} 0px 0px 2px 0px; display: none;`;
    }

    const style = document.createElement('style');
    style.innerHTML = `
        #filterBox > div:nth-child(2) > span:nth-child(2)::before,
        #filterBoxL > div:nth-child(2) > span:nth-child(2)::before {
            background: url('${filterIconUrl}') no-repeat center center;
            background-size: contain;
        }
        #refreshBtn > img,
        #refreshBtnL > img {
            content: url('${searchIconUrl}');
        }

        /* Gradio 4 */
        #filterBox > button:nth-child(2),
        #filterBoxL > button:nth-child(2) {
            background: url('${filterIconUrl}') no-repeat center center !important;
            background-size: 22px !important;
        }
        #filterBox > button:nth-child(2) > span,
        #filterBoxL > button:nth-child(2) > span {
            visibility: hidden;
        }
    `;
    document.head.appendChild(style);
}

// Creates a tooltip if the user wants to filter liked models without a personal API key
function createTooltip(element, hover_element, insertText) {
    if (element) {
        const tooltip = document.createElement('div');
        tooltip.className = 'browser_tooltip';
        tooltip.textContent = insertText;
        tooltip.style.cssText = 'display: none; text-align: center; white-space: pre;';

        hover_element.addEventListener('mouseover', () => {
            tooltip.style.display = 'block';
        });
        hover_element.addEventListener('mouseout', () => {
            tooltip.style.display = 'none';
        });
        element.appendChild(tooltip);
    }
}

// Function that closes filter dropdown if clicked outside the dropdown
function setupClickOutsideListener() {
    var filterBox = document.getElementById('filterBoxL') || document.getElementById('filterBox');
    var filterButton = filterBox.children[1];
    var dropDown = filterBox.getElementsByTagName('div')[2];

    function clickOutsideHandler(event) {
        var target = event.target;
        if (!filterBox.contains(target)) {
            if (!dropDown.contains(target)) {
                if (filterButton.className.endsWith('open')) {
                    filterButton.click();
                }
            }
        }
    }
    document.addEventListener('click', clickOutsideHandler);
}

// Create hyperlink in settings to CivitAI account settings
function createLink(infoElement) {
    const existingText = '(You can create your own API key in your CivitAI account settings, this required for some downloads. Requires UI reload)';
    const linkText = 'CivitAI account settings';

    const [textBefore, textAfter] = existingText.split(linkText);

    const link = document.createElement('a');
    link.textContent = linkText;
    link.href = 'https://civitai.com/user/account';
    link.target = '_blank';

    while (infoElement.firstChild) infoElement.removeChild(infoElement.firstChild);

    infoElement.appendChild(document.createTextNode(textBefore));
    infoElement.appendChild(link);
    infoElement.appendChild(document.createTextNode(textAfter));
}

// Create the accordion dropdown inside the settings tab
function createAccordion(containerDiv, subfolders, name, id_name) {
    if (containerDiv == null) {
        return;
    }
    var accordionContainer = document.createElement('div');
    accordionContainer.id = id_name;
    accordionContainer.className = 'settings-accordion';
    var toggleButton = document.createElement('button');
    toggleButton.id = 'accordionToggle';
    toggleButton.innerHTML = name + '<div style="transition: transform 0.15s; transform: rotate(90deg)">▼</div>';
    toggleButton.onclick = function () {
        accordionDiv.style.display = accordionDiv.style.display === 'none' ? 'block' : 'none';
        toggleButton.lastChild.style.transform = accordionDiv.style.display === 'none' ? 'rotate(90deg)' : 'rotate(0)';
    };

    accordionContainer.appendChild(toggleButton);
    var accordionDiv = document.createElement('div');
    accordionDiv.classList.add('accordion');
    if (subfolders && subfolders.length > 0) {
        accordionDiv.append(...subfolders);
    }

    accordionDiv.style.display = 'none'; // Initially hidden
    accordionContainer.appendChild(accordionDiv);
    containerDiv.appendChild(accordionContainer);
}

// Adds a button to the cards in txt2img and img2img
function createCivitAICardButtons() {
    const copyButton = document.querySelector('.copy-path-button');
    let fontSize = '1.8rem';
    if (!copyButton) {
        const editButton = document.querySelector('.edit-button');
        if (editButton) {
            const originalDisplay = editButton.parentElement.style.display;
            editButton.parentElement.style.display = 'flex';
            const editButtonBeforeStyle = window.getComputedStyle(editButton, ':before');
            fontSize = editButtonBeforeStyle.getPropertyValue('font-size');
            editButton.parentElement.style.display = originalDisplay;
        }
    }

    const checkForCardDivs = setInterval(() => {
        const cardDivs = document.querySelectorAll('.card');
        if (cardDivs.length > 0) {
            clearInterval(checkForCardDivs);

            cardDivs.forEach((cardDiv) => {
                const buttonRow = cardDiv.querySelector('.button-row');
                if (!buttonRow) return;

                buttonRow.addEventListener('click', function (event) {
                    event.stopPropagation();
                });

                if (!buttonRow.querySelector('.goto-civitbrowser.card-button')) {
                    const modelName = cardDiv.querySelector('.actions .name')?.textContent.trim();
                    if (!modelName) return;

                    const newDiv = document.createElement('div');
                    newDiv.className = 'goto-civitbrowser card-button';
                    const svgIcon = createSVGIcon(fontSize);
                    newDiv.appendChild(svgIcon);

                    newDiv.onclick = () => modelInfoPopUp(modelName, cardDiv.parentElement.id);
                    buttonRow.insertBefore(newDiv, buttonRow.firstChild);
                }
            });
        }
    }, 200);

    setTimeout(() => {
        clearInterval(checkForCardDivs);
    }, 5000);
}

function createSVGIcon(fontSize) {
    const svgIcon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svgIcon.setAttribute('width', fontSize);
    svgIcon.setAttribute('height', fontSize);
    svgIcon.setAttribute('viewBox', '75 85 350 350');
    svgIcon.setAttribute('fill', 'white');
    if (fontSize == '1.8rem') {
        svgIcon.setAttribute('style', 'margin-top: -2px');
    }
    svgIcon.innerHTML = `
        <path d="M 352.79 218.85 L 319.617 162.309 L 203.704 162.479 L 146.28 259.066 L 203.434 355.786 L 319.373 355.729 L 352.773 299.386 L 411.969 299.471 L 348.861 404.911 L 174.065 404.978 L 87.368 259.217 L 174.013 113.246 L 349.147 113.19 L 411.852 218.782 L 352.79 218.85 Z"/>
        <path d="M 304.771 334.364 L 213.9 334.429 L 169.607 259.146 L 214.095 183.864 L 305.132 183.907 L 330.489 227.825 L 311.786 259.115 L 330.315 290.655 Z M 278.045 290.682 L 259.294 259.18 L 278.106 227.488 L 240.603 227.366 L 221.983 259.128 L 240.451 291.026 Z"/>
    `;

    return svgIcon;
}

function addOnClickToButtons() {
    const tabs = ['img2img_extra_tabs', 'txt2img_extra_tabs'].map((id) => document.getElementById(id));
    const buttonIds = ['txt2img_checkpoints_extra_refresh', 'img2img_checkpoints_extra_refresh', 'txt2img_extra_refresh', 'img2img_extra_refresh'];

    buttonIds.forEach((buttonId) => {
        let button = document.getElementById(buttonId);
        if (button) {
            button.addEventListener('click', (event) => {
                createCivitAICardButtons(button);
            });
        }
    });

    tabs.forEach((tab) => {
        if (tab) {
            const buttons = tab.querySelectorAll('div > button:not(:first-child)');
            buttons.forEach((button) => {
                button.addEventListener('click', (event) => {
                    createCivitAICardButtons(button);
                });
            });
        }
    });
}

function modelInfoPopUp(modelName = null, content_type = null, no_message = false) {
    const sendToBrowserElement = gradioApp().querySelector('#setting_civitai_send_to_browser input');
    let sendToBrowser = false;
    if (sendToBrowserElement) {
        sendToBrowser = sendToBrowserElement.checked;
    }
    if (modelName) {
        select_model(modelName, null, true, content_type, sendToBrowser);
    }
    if (sendToBrowser) {
        const tabNav = document.querySelector('.tab-nav');
        const buttons = tabNav.querySelectorAll('button');
        for (const button of buttons) {
            if (button.textContent.includes('Browser+')) {
                button.click();

                const firstButton = document.querySelector('#tab_civitai_interface > div > div > div > button');
                if (firstButton) {
                    firstButton.click();
                }
            }
        }
    } else {
        createCivitaiOverlay(no_message);
    }
}

function createCivitaiOverlay(noMessage = false) {
    // Remove existing overlay if present
    const existingOverlay = document.querySelector('.civitai-overlay');
    if (existingOverlay) {
        existingOverlay.remove();
    }

    // Create overlay container
    const overlay = document.createElement('div');
    overlay.className = 'civitai-overlay';
    overlay.setAttribute('data-overlay-type', 'model-info');

    // Create inner content container
    const inner = document.createElement('div');
    inner.className = 'civitai-overlay-inner';

    // Create loading message if needed
    if (!noMessage) {
        const loadingMessage = document.createElement('div');
        loadingMessage.className = 'civitai-overlay-text';
        loadingMessage.textContent = 'Loading model info, please wait!';
        inner.appendChild(loadingMessage);
    }

    // Add event listeners
    overlay.addEventListener('click', handleOverlayClick);
    document.addEventListener('keydown', handleOverlayKeyPress);

    // Prevent body scroll
    document.body.style.overflow = 'hidden';

    // Append to DOM
    overlay.appendChild(inner);
    document.body.appendChild(overlay);

    // Trigger animation after DOM is ready
    setTimeout(() => {
        overlay.classList.add('show');
    }, 10);

    // Store reference for cleanup
    window.currentCivitaiOverlay = overlay;
}

// Handle overlay click events
function handleOverlayClick(event) {
    if (event.target.classList.contains('civitai-overlay')) {
        hideCivitaiOverlay();
    }
}
// Handle overlay keyboard events
function handleOverlayKeyPress(event) {
    if (event.key === 'Escape') {
        // Check if image viewer is open - if so, don't close overlay
        if (currentViewerOverlay && currentViewerOverlay.classList.contains('active')) {
            return; // Let image viewer handle ESC
        }
        hideCivitaiOverlay();
    }
}

function hideCivitaiOverlay() {
    const overlay = document.querySelector('.civitai-overlay');
    if (overlay) {
        // Start fade out animation
        overlay.classList.remove('show');

        // Wait for animation to complete before removing from DOM
        setTimeout(() => {
        // Remove event listeners
        overlay.removeEventListener('click', handleOverlayClick);
        document.removeEventListener('keydown', handleOverlayKeyPress);

        // Remove from DOM
        overlay.remove();

        // Restore body scroll
        document.body.style.overflow = 'auto';

        // Clear reference
        window.currentCivitaiOverlay = null;
        }, 300); // Match the CSS transition duration
    }
}

function inputHTMLPreviewContent(html_input) {
    const inner = document.querySelector('.civitai-overlay-inner');
    if (!inner) return;

    let startIndex = html_input.indexOf("'value': '");
    if (startIndex !== -1) {
        startIndex += "'value': '".length;
        let endIndex = html_input.indexOf(", 'placeholder'", startIndex);
        if (endIndex === -1) {
            endIndex = html_input.indexOf("', 'type': None,", startIndex);
        }
        if (endIndex !== -1) {
            let extractedText = html_input.substring(startIndex, endIndex);
            const modelIdNotFound = extractedText.includes('>Model ID not found.<br>The');

            // Clean up the HTML content
            extractedText = extractedText.replace(/\\n\s*</g, '<');
            extractedText = extractedText.replace(/\\n/g, ' ');
            extractedText = extractedText.replace(/\\t/g, '');
            extractedText = extractedText.replace(/\\'/g, "'");

            // Hide loading text
            const overlayText = document.querySelector('.civitai-overlay-text');
            if (overlayText) {
                overlayText.style.display = 'none';
            }

            // Create content container
            const modelInfo = document.createElement('div');
            modelInfo.innerHTML = extractedText;
            modelInfo.style.opacity = '0';
            modelInfo.style.transform = 'translateY(20px)';
            modelInfo.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
            inner.appendChild(modelInfo);

            // Allow inner container to expand to content height
            inner.style.height = 'auto';

            // Animate content appearance
            requestAnimationFrame(() => {
                modelInfo.style.opacity = '1';
                modelInfo.style.transform = 'translateY(0)';
            });

            // Initialize description toggle
            setTimeout(() => initDescriptionToggle('preview-'), 50);
        }
    }
}

function metaToTxt2Img(event, type, element) {
    const selection = window.getSelection();
    if (selection.toString().length > 0) {
        return;
    }
    const isAppend = event && event.shiftKey;
    const genButton = gradioApp().querySelector('#txt2img_extra_tabs > div > button');
    let input = element.querySelector('dd').textContent;
    let inf;
    if (input.endsWith(',')) {
        inf = input + ' ';
    } else {
        inf = input + ', ';
    }
    let is_positive = false;
    let is_negative = false;
    switch (type) {
        case 'Prompt':
            is_positive = true;
            break;
        case 'Negative prompt':
            inf = 'Negative prompt: ' + inf;
            is_negative = true;
            break;
        case 'Seed':
            inf = 'Seed: ' + inf;
            inf = inf + inf + inf;
            break;
        case 'Size':
            inf = 'Size: ' + inf;
            inf = inf + inf + inf;
            break;
        case 'Model':
            inf = 'Model: ' + inf;
            inf = inf + inf + inf;
            break;
        case 'Clip skip':
            inf = 'Clip skip: ' + inf;
            inf = inf + inf + inf;
            break;
        case 'Sampler':
            inf = 'Sampler: ' + inf;
            inf = inf + inf + inf;
            break;
        case 'Steps':
            inf = 'Steps: ' + inf;
            inf = inf + inf + inf;
            break;
        case 'CFG scale':
            inf = 'CFG scale: ' + inf;
            inf = inf + inf + inf;
            break;
    }
    const prompt = gradioApp().querySelector('#txt2img_prompt textarea');
    const neg_prompt = gradioApp().querySelector('#txt2img_neg_prompt textarea');
    const cfg_scale = gradioApp().querySelector('#txt2img_cfg_scale > div:nth-child(2) > div > input');
    let final = '';
    let cfg = 'CFG scale: ' + cfg_scale.value + ', ';
    let prompt_addon = cfg + cfg + cfg;
    if (is_positive) {
        if (isAppend) {
            const existing = prompt.value.trimEnd().replace(/,\s*$/, '');
            const combined = existing ? existing + ', ' + input : input;
            final = combined + '\nNegative prompt: ' + neg_prompt.value + '\n' + prompt_addon;
        } else {
            final = inf + '\nNegative prompt: ' + neg_prompt.value + '\n' + prompt_addon;
        }
    } else if (is_negative) {
        if (isAppend) {
            const existingNeg = neg_prompt.value.trimEnd().replace(/,\s*$/, '');
            const combinedNeg = existingNeg ? existingNeg + ', ' + input : input;
            final = prompt.value + '\nNegative prompt: ' + combinedNeg + '\n' + prompt_addon;
        } else {
            final = prompt.value + '\n' + inf + '\n' + prompt_addon;
        }
    } else {
        final = prompt.value + '\nNegative prompt: ' + neg_prompt.value + '\n' + inf;
    }
    genInfo_to_txt2img(final, false);
    hideCivitaiOverlay();
    sendClick(genButton);
}

// Creates a list of the selected models
var selectedModels = [];
var selectedTypes = [];
function multi_model_select(modelName, modelType, isChecked) {
    if (arguments.length === 0) {
        selectedModels = [];
        selectedTypes = [];
        return;
    }
    if (isChecked) {
        if (!selectedModels.includes(modelName)) {
            selectedModels.push(modelName);
        }
        selectedTypes.push(modelType);
    } else {
        var modelIndex = selectedModels.indexOf(modelName);
        if (modelIndex > -1) {
            selectedModels.splice(modelIndex, 1);
        }
        var typesIndex = selectedTypes.indexOf(modelType);
        if (typesIndex > -1) {
            selectedTypes.splice(typesIndex, 1);
        }
    }
    const selected_model_list = gradioApp().querySelector('#selected_model_list textarea');
    selected_model_list.value = JSON.stringify(selectedModels);

    const selected_type_list = gradioApp().querySelector('#selected_type_list textarea');
    selected_type_list.value = JSON.stringify(selectedTypes);

    updateInput(selected_model_list);
    updateInput(selected_type_list);
    syncUpdateBtn();
}

function sendClick(location) {
    const clickEvent = new MouseEvent('click', {
        view: window,
        bubbles: true,
        cancelable: true,
    });
    location.dispatchEvent(clickEvent);
}

let currentDlCancelled = false;

function cancelCurrentDl() {
    currentDlCancelled = true;
}

let allDlCancelled = false;

function cancelAllDl() {
    allDlCancelled = true;
}

function setSortable() {
    new Sortable(document.getElementById('queue_list'), {
        onEnd: function (evt) {
            const gradio_input = document.querySelector('#civitai_dl_list.prose').innerHTML;
            const gradio_html = gradioApp().querySelector('#queue_html_input textarea');
            let output = gradioApp().querySelector('#arrange_dl_id textarea');
            output.value = evt.item.getAttribute('dl_id') + '.' + evt.newIndex;
            updateInput(output);
            gradio_html.value = gradio_input;
            updateInput(gradio_html);
        },
    });
}

function cancelQueueDl() {
    const cancelBtn = gradioApp().querySelector('#html_cancel_input textarea');
    const randomNumber = Math.floor(Math.random() * 1000);
    const paddedNumber = String(randomNumber).padStart(3, '0');
    cancelBtn.value = paddedNumber;
    updateInput(cancelBtn);
    cancelBtn;
}

// Creates a setInterval-equivalent backed by a Web Worker so it runs
// unthrottled even when the browser tab is in the background.
function _createWorkerInterval(callback, ms) {
    const code = `var id;self.onmessage=function(e){if(e.data==='start'){id=setInterval(function(){self.postMessage('tick');},${ms});}else{clearInterval(id);self.close();}};`;
    const blob = new Blob([code], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    const worker = new Worker(url);
    worker.onmessage = function() { callback(); };
    worker.postMessage('start');
    worker._url = url;
    worker.stop = function() { worker.postMessage('stop'); URL.revokeObjectURL(url); };
    return worker;
}

function setDownloadProgressBar() {
    const gradio_html = gradioApp().querySelector('#queue_html_input textarea');
    let browserContainer = document.querySelector('#DownloadProgress');
    let browserProgress = browserContainer.querySelector('.progress-bar');
    if (!browserProgress || !browserProgress.style.width) {
        setTimeout(setDownloadProgressBar, 500);
        return;
    }

    let dlList = document.getElementById('civitai_dl_list');
    let nonQueue = dlList.querySelector('.civitai_nonqueue_list');
    let dlItem = dlList.querySelector('.civitai_dl_item');
    let dlBtn = dlItem.querySelector('.dl_action_btn > span');
    dlBtn.innerText = 'Cancel';
    dlBtn.setAttribute('onclick', 'cancelQueueDl()');
    let dlId = dlItem.getAttribute('dl_id');
    let selector = '.civitai_dl_item[dl_id="' + parseInt(dlId) + '"]';

    let dlProgressBar = null;
    let percentage = null;
    let dlText = null;

    nonQueue.appendChild(dlItem);

    const worker = _createWorkerInterval(() => {
        browserContainer = document.querySelector('#DownloadProgress');
        if (!browserContainer) {
            return; // User switched tab, container not in DOM
        }
        browserProgress = browserContainer.querySelector('.progress-bar');
        dlText = browserContainer.querySelector('.progress-level-inner');
        if (!dlText || !browserProgress) {
            return;
        }
        dlText = dlText.innerText;
        percentage = parseFloat(browserProgress.style.width);

        dlItem = dlList.querySelector(selector);
        if (!dlItem) {
            return;
        }
        dlProgressBar = dlItem.querySelector('.dl_progress_bar');
        if (!dlProgressBar) {
            return;
        }

        dlProgressBar.textContent = percentage.toFixed(1) + '%';
        dlProgressBar.style.width = percentage + '%';

        if (percentage >= 100) {
            worker.stop();
            dlBtn = dlItem.querySelector('.dl_action_btn > span');
            dlBtn.innerText = 'Remove';
            dlBtn.setAttribute('onclick', 'removeDlItem(' + parseInt(dlId) + ', this)');
            dlItem.className = 'civitai_dl_item_completed';
            dlProgressBar.textContent = 'Completed';
            dlProgressBar.style.width = '100%';
            const gradio_input = document.querySelector('#civitai_dl_list.prose').innerHTML;
            gradio_html.value = gradio_input;
            updateInput(gradio_html);
            return;
        }

        if (currentDlCancelled) {
            worker.stop();
            dlBtn = dlItem.querySelector('.dl_action_btn > span');
            dlBtn.innerText = 'Remove';
            dlBtn.setAttribute('onclick', 'removeDlItem(' + parseInt(dlId) + ', this)');
            currentDlCancelled = false;
            dlItem.className = 'civitai_dl_item_failed';
            dlProgressBar.textContent = 'Cancelled';
            dlProgressBar.style.width = '0%';
            const gradio_input = document.querySelector('#civitai_dl_list.prose').innerHTML;
            gradio_html.value = gradio_input;
            updateInput(gradio_html);
            return;
        } else if (allDlCancelled) {
            worker.stop();
            allDlCancelled = false;
            let dlItems = dlList.querySelectorAll('.civitai_dl_item');
            dlItems.forEach(function (item) {
                dlBtn = dlItem.querySelector('.dl_action_btn > span');
                dlBtn.innerText = 'Remove';
                dlBtn.setAttribute('onclick', 'removeDlItem(' + parseInt(dlId) + ', this)');
                dlProgressBar = item.querySelector('.dl_progress_bar');
                dlProgressBar.textContent = 'Cancelled';
                dlProgressBar.style.width = '0%';
                nonQueue.appendChild(item);
                item.className = 'civitai_dl_item_failed';
            });
            const gradio_input = document.querySelector('#civitai_dl_list.prose').innerHTML;
            gradio_html.value = gradio_input;
            updateInput(gradio_html);
            return;
        } else if (dlText.includes('Encountered an error during download of') || dlText.includes('not found on CivitAI servers') || dlText.includes('requires a personal CivitAI API to be downloaded')) {
            worker.stop();
            dlBtn = dlItem.querySelector('.dl_action_btn > span');
            dlBtn.innerText = 'Remove';
            dlBtn.setAttribute('onclick', 'removeDlItem(' + parseInt(dlId) + ', this)');
            dlItem.className = 'civitai_dl_item_failed';
            dlProgressBar.textContent = 'Failed';
            dlProgressBar.style.width = '0%';
            const gradio_input = document.querySelector('#civitai_dl_list.prose').innerHTML;
            gradio_html.value = gradio_input;
            updateInput(gradio_html);
            return;
        }
    });
}

// === Queue Restore Banner ===

function initRestoreBanner(json) {
    const banner = gradioApp().querySelector('#restore_banner');
    if (!banner) return;
    if (!json || json.trim() === '') { banner.innerHTML = ''; return; }
    let items;
    try { items = JSON.parse(json); } catch (e) { return; }
    if (!items || items.length === 0) { banner.innerHTML = ''; return; }
    const count = items.length;
    const names = items.slice(0, 3).map(i => `<b>${i.model_name}</b>`).join(', ');
    const more = count > 3 ? ` and ${count - 3} more` : '';
    banner.innerHTML = `
        <div style="background:#1a3a5c;border:1px solid #2d6a9f;border-radius:8px;
                    padding:12px 16px;margin:8px 0;display:flex;align-items:center;
                    gap:12px;flex-wrap:wrap;">
            <span style="flex:1;min-width:200px;">
                🔄 <b>${count} download${count > 1 ? 's' : ''} need attention.</b> ${names}${more}
            </span>
            <button onclick="triggerRestoreQueue()"
                style="background:#2d6a9f;color:white;border:none;border-radius:6px;
                       padding:6px 14px;cursor:pointer;font-size:14px;">↺ Restore Queue</button>
            <button onclick="triggerDismissRestore()"
                style="background:transparent;color:#aaa;border:1px solid #555;
                       border-radius:6px;padding:6px 14px;cursor:pointer;font-size:14px;">✕ Dismiss</button>
        </div>`;
}

function triggerRestoreQueue() {
    const trigger = gradioApp().querySelector('#restore_action_trigger textarea');
    if (!trigger) return;
    trigger.value = String(Date.now());
    updateInput(trigger);
    const banner = gradioApp().querySelector('#restore_banner');
    if (banner) banner.innerHTML = '';
}

function triggerDismissRestore() {
    const trigger = gradioApp().querySelector('#dismiss_restore_trigger textarea');
    if (!trigger) return;
    trigger.value = '1';
    updateInput(trigger);
    const banner = gradioApp().querySelector('#restore_banner');
    if (banner) banner.innerHTML = '';
}

function removeDlItem(dl_id, element) {
    const gradio_html = gradioApp().querySelector('#queue_html_input textarea');
    const output = gradioApp().querySelector('#remove_dl_id textarea');
    var dl_item = element.parentNode.parentNode;
    dl_item.parentNode.removeChild(dl_item);
    output.value = dl_id;
    updateInput(output);

    const gradio_input = document.querySelector('#civitai_dl_list.prose').innerHTML;
    gradio_html.value = gradio_input;
    updateInput(gradio_html);
}

// Selects all models
function selectAllModels() {
    const checkboxes = Array.from(document.querySelectorAll('.model-checkbox'));
    const allChecked = checkboxes.every((checkbox) => checkbox.checked);
    const allUnchecked = checkboxes.every((checkbox) => !checkbox.checked);
    if (allChecked || allUnchecked) {
        checkboxes.forEach(sendClick);
    } else {
        checkboxes.filter((checkbox) => !checkbox.checked).forEach(sendClick);
    }
}

// Deselects all models
function deselectAllModels() {
    setTimeout(() => {
        const checkboxes = Array.from(document.querySelectorAll('.model-checkbox'));
        checkboxes.filter((checkbox) => checkbox.checked).forEach(sendClick);
    }, 1000);
}

// Sends Image URL to Python to pull generation info
function sendImgUrl(image_url) {
    const randomNumber = Math.floor(Math.random() * 1000);
    const genButton = gradioApp().querySelector('#txt2img_extra_tabs > div > button');
    const paddedNumber = String(randomNumber).padStart(3, '0');
    const input = gradioApp().querySelector('#civitai_text2img_input textarea');
    input.value = paddedNumber + '.' + image_url;
    updateInput(input);
    hideCivitaiOverlay();
    sendClick(genButton);
}

// Triggers a browser download for text content using a temporary Blob URL.
// Called by the export_csv_output / export_json_output change events.
function downloadBlobFile(content, filename, mimeType) {
    if (!content || !content.trim()) return;  // no data yet — ignore reset to ''
    const blob = new Blob([content], { type: mimeType });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    // Reset the hidden textbox so the next click fires a fresh change event
    const outputId = mimeType === 'text/csv' ? 'export_csv_output' : 'export_json_output';
    const box = gradioApp().querySelector(`#${outputId} textarea`);
    if (box) {
        const nativeSet = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
        nativeSet.call(box, '');
        box.dispatchEvent(new Event('input', { bubbles: true }));
    }
}

// Sends trained tags (trigger words) to txt2img prompt
// Shift+click appends to existing prompt; regular click replaces
function copyTriggerWord(text, btn) {
    navigator.clipboard.writeText(text).then(() => {
        const orig = btn.textContent;
        btn.textContent = '✓';
        setTimeout(() => { btn.textContent = orig; }, 1500);
    }).catch(() => {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch (_) {}
        document.body.removeChild(ta);
        const orig = btn.textContent;
        btn.textContent = '✓';
        setTimeout(() => { btn.textContent = orig; }, 1500);
    });
}

function sendTagsToPrompt(tags) {
    if (!tags || !tags.trim()) return;
    const genButton = gradioApp().querySelector('#txt2img_extra_tabs > div > button');
    const prompt = gradioApp().querySelector('#txt2img_prompt textarea');
    const neg_prompt = gradioApp().querySelector('#txt2img_neg_prompt textarea');
    const cfg_scale = gradioApp().querySelector('#txt2img_cfg_scale > div:nth-child(2) > div > input');
    const cfg = 'CFG scale: ' + cfg_scale.value + ', ';
    const prompt_addon = cfg + cfg + cfg;
    const cleanTags = tags.trimEnd().replace(/,\s*$/, '');
    const existing = (prompt.value || '').trimEnd().replace(/,\s*$/, '');
    const combined = existing ? existing + ', ' + cleanTags : cleanTags;
    const final = combined + '\nNegative prompt: ' + (neg_prompt.value || '') + '\n' + prompt_addon;
    genInfo_to_txt2img(final, false);
    sendClick(genButton);
    hideCivitaiOverlay();
}

// Sends txt2img info to txt2img tab
function genInfo_to_txt2img(genInfo, do_slice = true) {
    let insert = gradioApp().querySelector('#txt2img_prompt textarea');
    let pasteButton = gradioApp().querySelector('#paste');
    if (genInfo) {
        insert.value = do_slice ? genInfo.slice(5) : genInfo;
        // updateInput notifies Gradio 4's React layer that the value changed
        updateInput(insert);
        // Must use .click() or MouseEvent — generic Event('click') is silently
        // ignored by Gradio 4's button handler in some browser/focus states
        pasteButton.click();
    }
}

// Hide installed models
function hideInstalled(toggleValue) {
    const modelList = document.querySelectorAll('.column.civmodellist > .civmodelcardinstalled');
    modelList.forEach((item) => {
        item.style.display = toggleValue ? 'none' : 'block';
    });
}

// === Creator Management (ban / favorite) ===
let _bannedCreators = [];

// Called when a new model list loads — sync banned list then apply filter
function initBannedCreators(listStr, checked) {
    _bannedCreators = listStr ? listStr.split(',').map(s => s.trim()).filter(Boolean) : [];
    hideBannedCreators(checked);
}

// Called after ban/fav button actions — refresh list and re-apply filter
function refreshBannedCreators(listStr, checked) {
    _bannedCreators = listStr ? listStr.split(',').map(s => s.trim()).filter(Boolean) : [];
    hideBannedCreators(checked);
}

// Show or hide cards of banned creators
function hideBannedCreators(checked) {
    document.querySelectorAll('.civmodelcard[data-creator]').forEach(card => {
        const creator = card.getAttribute('data-creator');
        const isBanned = creator && _bannedCreators.includes(creator);
        if (isBanned) {
            card.style.display = checked ? 'none' : '';
        }
    });
}

// Re-apply hideInstalled and banned-creator filters by reading current toggle states.
// Called from MutationObserver when new cards are injected into the DOM.
function reapplyFilters() {
    const hideInstalledToggle =
        gradioApp().querySelector('#toggle5 input[type="checkbox"]') ||
        gradioApp().querySelector('#toggle5L input[type="checkbox"]');
    if (hideInstalledToggle) {
        hideInstalled(hideInstalledToggle.checked);
    }

    const hideBannedToggle =
        gradioApp().querySelector('#hideBannedCreators input[type="checkbox"]');
    const bannedListInput =
        gradioApp().querySelector('#banned_creators_list');
    if (hideBannedToggle && bannedListInput) {
        refreshBannedCreators(bannedListInput.value, hideBannedToggle.checked);
    }
}

// Toggle description visibility
function toggleDescription(prefix = '') {
    const content = document.getElementById(prefix + 'description-content');
    const overlay = document.getElementById(prefix + 'description-overlay');
    const button = document.getElementById(prefix + 'description-toggle-btn');

    if (!content || !overlay || !button) return;

    const isExpanded = content.classList.contains('expanded');

    if (isExpanded) {
        // Collapse - animate back to 400px
        content.style.maxHeight = '400px';
        content.classList.remove('expanded');
        overlay.classList.remove('hidden');
        button.textContent = 'Show More';
    } else {
        // Expand - calculate full height and animate to it
        const scrollHeight = content.scrollHeight;
        content.style.maxHeight = scrollHeight + 'px';
        content.classList.add('expanded');
        overlay.classList.add('hidden');
        button.textContent = 'Show Less';
    }
}

// Initialize description toggle functionality
function initDescriptionToggle(prefix = '') {
    const content = document.getElementById(prefix + 'description-content');
    const overlay = document.getElementById(prefix + 'description-overlay');
    const button = document.getElementById(prefix + 'description-toggle-btn');

    if (!content || !overlay || !button) return;

    // Reset styles first
    content.style.maxHeight = '';
    content.classList.remove('expanded');
    overlay.classList.remove('hidden');
    button.classList.remove('hidden');

    // Check if content height exceeds 400px
    const scrollHeight = content.scrollHeight;

    // If content is less than or equal to 400px, hide toggle elements
    if (scrollHeight <= 400) {
        overlay.classList.add('hidden');
        button.classList.add('hidden');
        content.style.maxHeight = 'none';
    } else {
        // Set initial collapsed state
        content.style.maxHeight = '400px';
        button.textContent = 'Show More';
    }
}

function submitNewSubfolder(subfolderId, subfolderValue) {
    const output = gradioApp().querySelector('#create_subfolder textarea');
    output.value = subfolderId + '.add.' + subfolderValue;
    updateInput(output);
}

function deleteSubfolder(subfolderId) {
    const output = gradioApp().querySelector('#create_subfolder textarea');
    output.value = subfolderId + '.delete.';
    updateInput(output);
}

function createCustomSubfolder(subfolderDiv, subfolderId, subfolderValue) {
    if (typeof subfolderId === 'undefined') {
        console.error('subfolderId is required.');
        return;
    }

    const newContainerDiv = document.createElement('div');
    newContainerDiv.classList.add('svelte-1f354aw', 'container', 'CivitDefaultSubfolder');
    newContainerDiv.style.display = 'flex';
    newContainerDiv.style.alignItems = 'center';

    newContainerDiv.setAttribute('subfolder_id', subfolderId);

    const newTextArea = document.createElement('textarea');
    newTextArea.setAttribute('data-testid', 'textbox');
    newTextArea.classList.add('scroll-hide', 'svelte-1f354aw');
    newTextArea.setAttribute('dir', 'ltr');
    newTextArea.setAttribute('placeholder', '{BASEMODEL}/{NSFW}/{AUTHOR}/{MODELNAME}/{MODELID}/{VERSIONNAME}/{VERSIONID}');
    newTextArea.setAttribute('rows', '1');
    newTextArea.style.overflowY = 'scroll';
    newTextArea.style.height = '42px';
    newTextArea.style.flex = '1';

    if (typeof subfolderValue !== 'undefined') {
        newTextArea.value = subfolderValue;
    }

    newTextArea.addEventListener('keydown', function (event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            submitNewSubfolder(subfolderId, newTextArea.value);
        }
    });

    const saveButton = document.createElement('button');
    saveButton.textContent = 'Save';
    saveButton.classList.add('save-button', 'lg', 'primary', 'gradio-button', 'svelte-cmf5ev');
    saveButton.setAttribute('title', '');
    saveButton.style.marginRight = '10px';
    saveButton.addEventListener('click', function () {
        submitNewSubfolder(subfolderId, newTextArea.value);
    });

    const deleteButton = document.createElement('button');
    deleteButton.textContent = 'Delete';
    deleteButton.classList.add('delete-button', 'lg', 'primary', 'gradio-button', 'svelte-cmf5ev');
    deleteButton.style.marginRight = '10px';
    deleteButton.addEventListener('click', function () {
        deleteSubfolder(subfolderId);
        newContainerDiv.remove();
    });

    newContainerDiv.appendChild(deleteButton);
    newContainerDiv.appendChild(saveButton);
    newContainerDiv.appendChild(newTextArea);

    subfolderDiv.appendChild(newContainerDiv);
}

function insertExistingSubfolders(input) {
    const subfolder = document.querySelectorAll('civitai-custom-subfolder-div');
    createCustomSubfolder(subfolder, Id, Value);
}

function createSubfolderButton() {
    const subfolderParent = document.getElementById('create-sub-accordion');
    const subfolderDiv = subfolderParent.querySelector('.accordion');

    const subfolder = document.createElement('div');
    subfolder.classList.add('flex-column-layout', 'civitai-custom-subfolder-div');

    const customSubfoldersList = document.querySelector('#custom_subfolders_list');
    const textarea = customSubfoldersList.querySelector('textarea');
    const subfoldersString = textarea ? textarea.value : '';

    const subfoldersArray = subfoldersString.split('␞␞');

    for (let i = 0; i < subfoldersArray.length; i += 2) {
        const subfolderId = subfoldersArray[i];
        const subfolderValue = subfoldersArray[i + 1];

        createCustomSubfolder(subfolder, subfolderId, subfolderValue);
    }

    const buttonContainer = document.createElement('div');
    buttonContainer.classList.add('sub-folder-button-container');
    buttonContainer.style.display = 'flex';
    buttonContainer.style.gap = '10px';

    const optionsDiv = document.createElement('div');
    optionsDiv.classList.add('placeholder-options-container');
    optionsDiv.style.display = 'flex';
    optionsDiv.style.justifyContent = 'center';

    const plusButton = document.createElement('button');
    plusButton.textContent = 'Create new default sub folder entry';
    plusButton.classList.add('plus-button', 'lg', 'primary', 'gradio-button', 'svelte-cmf5ev');
    plusButton.style.marginTop = '10px';
    plusButton.addEventListener('click', function () {
        const existingSubfolderDivs = document.querySelectorAll('div.CivitDefaultSubfolder');
        let highestSubfolderId = 0;

        existingSubfolderDivs.forEach((div) => {
            const subfolderId = parseInt(div.getAttribute('subfolder_id'), 10);
            if (subfolderId > highestSubfolderId) {
                highestSubfolderId = subfolderId;
            }
        });

        const newSubfolderId = highestSubfolderId + 1;
        createCustomSubfolder(subfolder, newSubfolderId);
    });

    // Create the guide button
    const guide_html = `
    <div style="text-align: center;">
        <div>These options can be used to add sub-folder options.</div>
        <div>There are a few placeholders you can use which will be automatically replaced with the selected model's information:</div>
        <div>‎</div>
        <div>{BASEMODEL}: Replaced with the base model name.</div>
        <div>{NSFW}: Creates a folder named "nsfw", folder will not be created if model is sfw.</div>
        <div>{AUTHOR}: Replaced with the author of the model.</div>
        <div>{MODELNAME}: Replaced with the name of the model.</div>
        <div>{MODELID}: Replaced with the unique ID of the model.</div>
        <div>{VERSIONNAME}: Replaced with the version name of the model.</div>
        <div>{VERSIONID}: Replaced with the unique ID of the model version.</div>
        <div>‎</div>
        <div>For example, if I select a model called 'ReV Animated'</div>
        <div>and it's version is called 'V2 Rebirth' then the following:</div>
        <div>{MODELNAME}/{VERSIONNAME}</div>
        <div>Will be replaced with:</div>
        <div>ReV Animated/V2 Rebirth</div>
        <div>‎</div>
        <div>Always use '/' as a seperator, regardless of your OS</div>
    </div>
    `;
    const guideButton = document.createElement('button');
    guideButton.textContent = 'Guide';
    guideButton.classList.add('guide-button', 'lg', 'primary', 'gradio-button', 'svelte-cmf5ev');
    guideButton.style.marginTop = '10px';
    guideButton.addEventListener('click', function () {
        modelInfoPopUp(null, null, true);
        insertGuideMessage(guide_html);
    });

    const optionsText = document.createElement('span');
    optionsText.textContent = 'Available options: {BASEMODEL} {NSFW} {AUTHOR} {MODELNAME} {MODELID} {VERSIONNAME} {VERSIONID}';

    // Append buttons to the container
    buttonContainer.appendChild(guideButton);
    buttonContainer.appendChild(plusButton);

    optionsDiv.appendChild(optionsText);

    subfolder.insertBefore(optionsDiv, subfolder.firstChild);
    subfolder.insertBefore(buttonContainer, subfolder.firstChild);
    subfolderDiv.appendChild(subfolder);
}

function insertGuideMessage(html_input) {
    const overlayContainer = document.querySelector('.civitai-overlay-inner');
    if (overlayContainer) {
        const guideHtml = document.createElement('div');
        guideHtml.innerHTML = html_input;
        while (guideHtml.firstChild) {
            overlayContainer.appendChild(guideHtml.firstChild);
        }
    }
}

// === ANXETY EDITs ===
// Runs all functions when the page is fully loaded
function onPageLoad() {
    updateSVGIcons();

    let subfolderDiv = document.querySelector('#settings_civitai_browser_plus > div > div');
    let downloadDiv = document.querySelector('#settings_civitai_browser_download > div > div');
    let settingsDiv = document.querySelector('#settings_civitai_browser > div > div');

    if (subfolderDiv || downloadDiv) {
        let div = subfolderDiv || downloadDiv;
        let subfolders = div.querySelectorAll("[id$='subfolder']");
        createAccordion(div, subfolders, 'Default sub folders', 'default-sub-accordion');
        createAccordion(div, null, 'Create sub folder entries', 'create-sub-accordion');
        createSubfolderButton();
    }

    if (subfolderDiv || settingsDiv) {
        let div = subfolderDiv || settingsDiv;
        let proxy = div.querySelectorAll("[id$='proxy']");
        createAccordion(div, proxy, 'Proxy options', 'proxy-accordion');
    }

    let toggle4L = document.getElementById('toggle4L');
    let toggle4 = document.getElementById('toggle4');
    let hash_toggle_hover = document.querySelector('#skip_hash_toggle > label');
    let hash_toggle = document.querySelector('#skip_hash_toggle');
    let do_html_gen_hover = document.querySelector('#do_html_gen > label');
    let do_html_gen = document.querySelector('#do_html_gen');

    if (toggle4L || toggle4) {
        let like_toggle = toggle4L || toggle4;
        let insertText = 'Requires an API Key\nConfigurable in CivitAI settings tab';
        createTooltip(like_toggle, like_toggle, insertText);
    }

    if (hash_toggle) {
        let insertText =
            'This option generates unique hashes for models that were not downloaded with this extension.\nA hash is required for any of the options below to work, a model with no hash will be skipped.\nInitial hash generation is a one-time process per file.';
        createTooltip(hash_toggle, hash_toggle_hover, insertText);
    }

    if (do_html_gen) {
        let insertText =
            'This option requires the "Save HTML file when saving model" option to be enabled.\nYou can find this setting in the CivitAI Browser+ settings under Downloads section.';
        createTooltip(do_html_gen, do_html_gen_hover, insertText);
    }

    addOnClickToButtons();
    createCivitAICardButtons();
    adjustFilterBoxAndButtons();
    setupClickOutsideListener();
}

onUiLoaded(onPageLoad);

function checkSettingsLoad() {
    const divElement = gradioApp().querySelector('#setting_custom_api_key');
    const infoElement = divElement?.querySelector('.info');
    if (!infoElement) {
        return;
    }
    clearInterval(settingsLoadInterval);
    createLink(infoElement);
}
let settingsLoadInterval = setInterval(checkSettingsLoad, 1000);

// === Simple Image Viewer ===
// Global viewer state
let currentViewerOverlay = null;
let viewerEventListeners = [];

// Create viewer overlay dynamically
function createViewerOverlay() {
    // Remove existing viewer if it exists
    const existingViewer = document.getElementById('image-viewer-overlay');
    if (existingViewer) {
        existingViewer.remove();
    }

    // Create new viewer overlay
    const overlay = document.createElement('div');
    overlay.id = 'image-viewer-overlay';
    overlay.className = 'viewer-overlay';

    const content = document.createElement('div');
    content.className = 'viewer-content';

    const image = document.createElement('img');
    image.id = 'viewer-image';
    image.className = 'viewer-media';
    image.alt = '';

    const video = document.createElement('video');
    video.id = 'viewer-video';
    video.className = 'viewer-media';
    video.controls = true;
    video.muted = true;
    video.style.display = 'none';

    const source = document.createElement('source');
    source.type = 'video/mp4';
    video.appendChild(source);

    content.appendChild(image);
    content.appendChild(video);
    overlay.appendChild(content);

    // Add to body
    document.body.appendChild(overlay);

    return overlay;
}

// Open image viewer overlay
function openImageViewer(mediaUrl, mediaType) {
    // Create or get viewer overlay
    let overlay = document.getElementById('image-viewer-overlay');
    if (!overlay) {
        overlay = createViewerOverlay();
    }

    const viewerImage = document.getElementById('viewer-image');
    const viewerVideo = document.getElementById('viewer-video');

    if (!overlay || !viewerImage || !viewerVideo) return;

    // Setup media element
    if (mediaType === 'video') {
        viewerImage.style.display = 'none';
        viewerVideo.style.display = 'block';
        const source = viewerVideo.querySelector('source');
        if (source) {
            source.src = mediaUrl;
            viewerVideo.load();
        }
    } else {
        viewerVideo.style.display = 'none';
        viewerImage.style.display = 'block';
        viewerImage.src = mediaUrl;
    }

    // Position overlay to cover entire viewport with proper centering
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100vw';
    overlay.style.height = '100vh';
    overlay.style.zIndex = '99999';
    overlay.style.display = 'flex';
    overlay.style.alignItems = 'center';
    overlay.style.justifyContent = 'center';

    // Show overlay with animation
    overlay.style.display = 'flex';
    overlay.classList.remove('closing');
    requestAnimationFrame(() => {
        overlay.classList.add('active');
    });

    // Prevent body scroll
    document.body.style.overflow = 'hidden';

    // Setup event listeners
    setupViewerEventListeners(overlay);

    currentViewerOverlay = overlay;
}

// Setup event listeners for viewer
function setupViewerEventListeners(overlay) {
    // Remove existing listeners
    cleanupViewerEventListeners();

    const viewerImage = document.getElementById('viewer-image');
    const viewerVideo = document.getElementById('viewer-video');

    // Keyboard handler
    const keyHandler = (e) => {
        if (e.key === 'Escape') {
            closeImageViewer();
            e.stopPropagation();
            e.preventDefault();
        }
    };

    // Overlay click handler
    const overlayHandler = (e) => {
        if (e.target.classList.contains('viewer-overlay') || e.target.classList.contains('viewer-content')) {
            closeImageViewer();
        }
    };

    // Media click handler
    const mediaHandler = (e) => {
        e.stopPropagation();
    };

    // Add listeners
    document.addEventListener('keydown', keyHandler);
    overlay.addEventListener('click', overlayHandler);
    if (viewerImage) viewerImage.addEventListener('click', mediaHandler);
    if (viewerVideo) viewerVideo.addEventListener('click', mediaHandler);

    // Store references for cleanup
    viewerEventListeners = [
        {element: document, event: 'keydown', handler: keyHandler},
        {element: overlay, event: 'click', handler: overlayHandler},
        {element: viewerImage, event: 'click', handler: mediaHandler},
        {element: viewerVideo, event: 'click', handler: mediaHandler},
    ];
}

// Cleanup event listeners
function cleanupViewerEventListeners() {
    viewerEventListeners.forEach(({element, event, handler}) => {
        if (element && handler) {
            element.removeEventListener(event, handler);
        }
    });
    viewerEventListeners = [];
}

// Close image viewer overlay
function closeImageViewer() {
    const overlay = document.getElementById('image-viewer-overlay');
    if (!overlay || !overlay.classList.contains('active')) return;

    // Add closing animation
    overlay.classList.add('closing');
    overlay.classList.remove('active');

    setTimeout(() => {
        overlay.style.display = 'none';
        overlay.classList.remove('closing');

        // Restore body scroll only if civitai overlay is not open
        if (!document.querySelector('.civitai-overlay')) {
            document.body.style.overflow = 'auto';
        }

        // Cleanup event listeners
        cleanupViewerEventListeners();

        currentViewerOverlay = null;
    }, 300);
}

// Global flag to track if viewer is initialized
let viewerInitialized = false;
let previewMediaObserver = null;

// Handle click on preview media - lazy initialization
function handlePreviewMediaClick(e) {
    e.preventDefault();
    e.stopPropagation();

    // Initialize viewer on first click if not already done
    if (!viewerInitialized) {
        initializeImageViewer();
        viewerInitialized = true;
    }

    const element = e.target;
    const isVideo = element.tagName.toLowerCase() === 'video';
    const mediaUrl = isVideo ? element.querySelector('source')?.src || element.src : element.src;

    if (mediaUrl) {
        openImageViewer(mediaUrl, isVideo ? 'video' : 'image');
    }
}

// Initialize image viewer for preview media elements (lazy initialization)
function initializeImageViewer() {
    // Add click handlers to existing preview media elements
    const previewMediaElements = document.querySelectorAll('.preview-media');
    previewMediaElements.forEach((element) => {
        element.addEventListener('click', handlePreviewMediaClick);
    });

    // Set up MutationObserver for dynamically added elements
    if (!previewMediaObserver) {
        previewMediaObserver = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        // Check if the added node is a preview media element
                        if (node.classList && node.classList.contains('preview-media')) {
                            node.addEventListener('click', handlePreviewMediaClick);
                        }
                        // Check for preview media elements within the added node
                        const previewElements = node.querySelectorAll && node.querySelectorAll('.preview-media');
                        if (previewElements) {
                            previewElements.forEach((element) => {
                                element.addEventListener('click', handlePreviewMediaClick);
                            });
                        }
                    }
                });
            });
        });

        // Start observing
        previewMediaObserver.observe(document.body, {
            childList: true,
            subtree: true,
        });
    }
}

// Delete installed model with confirmation
function deleteInstalledModel(event, modelString, sha256, installedCount = 1) {
    // Stop event propagation to prevent card selection
    event.stopPropagation();
    event.preventDefault();

    const installedTotal = Number(installedCount || 0);
    if (installedTotal > 1) {
        select_model(modelString, null);
        alert(
            `This model has ${installedTotal} installed versions.\n\n` +
            'For safety, quick delete is disabled.\n' +
            'Please choose the exact [Installed] version in the Browser panel and use "Delete model" there.'
        );
        return;
    }
    
    if (!sha256) {
        alert('Error: No SHA256 hash available for this model. Cannot delete.');
        return;
    }
    
    // Strip the trailing " (id)" part for display only
    const displayName = modelString.replace(/\s*\(\d+\)\s*$/, '');
    
    // Show confirmation dialog
    const confirmMessage = `Are you sure you want to delete "${displayName}"?\n\nThis will move the model and its associated files to the trash.`;
    if (!confirm(confirmMessage)) {
        return;
    }
    
    // Find the delete trigger input element (presence check — ensures UI is loaded)
    const deleteFinishInput = gradioApp().querySelector('#delete_finish textarea');
    if (!deleteFinishInput) {
        alert('Error: Delete function not available.');
        console.error('Could not find #delete_finish element');
        return;
    }
    
    // Find the SHA256 input element
    const sha256Input = gradioApp().querySelector('#sha256 textarea');
    if (!sha256Input) {
        alert('Error: SHA256 input not found.');
        console.error('Could not find #sha256 element');
        return;
    }
    
    // Set SHA256 value
    sha256Input.value = sha256;
    updateInput(sha256Input);
    
    // Wait a bit for the SHA256 to be set, then trigger delete
    setTimeout(() => {
        // Trigger delete by updating delete_trigger_btn click
        const deleteButton = gradioApp().querySelector('#delete_trigger_btn');
        if (deleteButton) {
            deleteButton.click();
            // After deletion completes, update the card visually (remove installed state).
            // Skip pressRefresh() fallback to avoid expensive full-page API re-fetch.
            setTimeout(() => updateCard(modelString + '.None', false), 500);
        } else {
            console.error('Could not find #delete_trigger_btn element');
            alert('Error: Delete button not found. Please try using the delete button in the model details panel.');
        }
    }, 100);
}


// ── Update Mode ──────────────────────────────────────────────────────────────

/**
 * Trigger the Python backend to enqueue ALL models from gl.update_items.
 * Called by the "⬆️ Update All" button in the update mode action bar.
 */
function updateAllModels() {
    const trigger = gradioApp().querySelector('#update_all_trigger textarea');
    if (!trigger) return;
    trigger.value = String(Date.now());
    updateInput(trigger);
}

/**
 * Called by the "Update All / Update Selected" button.
 * If any update-grid checkboxes are checked, updates only those; otherwise updates all.
 */
function updateOrSelectedModels() {
    if (selectedModels.length > 0) {
        const trigger = gradioApp().querySelector('#update_selected_trigger textarea');
        if (!trigger) return;
        trigger.value = JSON.stringify(selectedModels);
        updateInput(trigger);
        // Visual feedback: dim checked cards
        gradioApp().querySelectorAll('.model-checkbox:checked').forEach(cb => {
            const card = cb.closest('.civmodelcard');
            if (card) card.style.opacity = '0.4';
        });
    } else {
        updateAllModels();
    }
}

/**
 * Syncs the Update All/Selected button label with the current selection count.
 */
function syncUpdateBtn() {
    const btn = gradioApp().querySelector('#civupdate-update-btn');
    if (!btn) return;
    const n     = selectedModels.length;
    const total = gradioApp().querySelectorAll('.model-checkbox').length;
    btn.textContent = n > 0
        ? `\u2b06\ufe0f Update Selected (${n})`
        : `\u2b06\ufe0f Update All (${total})`;
}

/**
 * Trigger the Python backend to enqueue a SINGLE model update.
 * Called by the ⬆ button on an individual update card.
 * Collects checked version IDs from the card's version checkboxes.
 * @param {string|number} modelId  - the CivitAI model id
 * @param {string}        family   - the installed family (e.g. 'PONY', 'IL', '')
 */
function updateSingleModel(modelId, family) {
    const card = gradioApp().querySelector(
        `.update-mode-card[data-model-id="${modelId}"][data-family="${(family || '').toUpperCase()}"]`
    );

    let checkedVersions = [];
    if (card) {
        checkedVersions = Array.from(
            card.querySelectorAll('.ver-checkbox:checked')
        ).map(cb => parseInt(cb.dataset.verId, 10));
    }

    const trigger = gradioApp().querySelector('#update_single_trigger textarea');
    if (!trigger) return;

    // Format: model_id|family|json_array_of_ver_ids
    // If no versions checked, sends empty array → Python falls back to auto-resolution
    trigger.value = `${modelId}|${family || ''}|${JSON.stringify(checkedVersions)}`;
    updateInput(trigger);

    // Visual feedback: dim the card
    if (card) card.style.opacity = '0.4';
}

/**
 * Exit Update Mode: clear the banner and tell Python to reset the page state.
 */
function exitUpdateMode() {
    const trigger = gradioApp().querySelector('#exit_update_mode_trigger textarea');
    if (!trigger) return;
    trigger.value = String(Date.now());
    updateInput(trigger);
    // Immediately clear the banner
    const banner = gradioApp().querySelector('#update_mode_banner');
    if (banner) banner.innerHTML = '';
}