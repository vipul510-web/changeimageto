// Background script for ChangeImageTo Chrome Extension
// Handles context menus, API communication, and extension lifecycle

(function() {
    'use strict';
    
    // Configuration
    const BACKEND_URL = 'https://bgremover-backend-121350814881.us-central1.run.app';
    
    // Initialize extension
    chrome.runtime.onInstalled.addListener(() => {
        console.log('ChangeImageTo extension installed');
        setupContextMenus();
    });
    
    // Setup context menus
    function setupContextMenus() {
        // Remove existing menus
        chrome.contextMenus.removeAll(() => {
            // Main menu for images
            chrome.contextMenus.create({
                id: 'changeimageto-main',
                title: 'Edit with ChangeImageTo',
                contexts: ['image'],
                documentUrlPatterns: ['<all_urls>']
            });
            
            // Submenu for image operations
            chrome.contextMenus.create({
                id: 'remove-background',
                parentId: 'changeimageto-main',
                title: 'Remove Background',
                contexts: ['image']
            });
            
            chrome.contextMenus.create({
                id: 'blur-background',
                parentId: 'changeimageto-main',
                title: 'Blur Background',
                contexts: ['image']
            });
            
            chrome.contextMenus.create({
                id: 'enhance-image',
                parentId: 'changeimageto-main',
                title: 'Enhance Image',
                contexts: ['image']
            });
            
            chrome.contextMenus.create({
                id: 'upscale-image',
                parentId: 'changeimageto-main',
                title: 'Upscale Image',
                contexts: ['image']
            });
            
            // Separator
            chrome.contextMenus.create({
                id: 'separator1',
                parentId: 'changeimageto-main',
                type: 'separator',
                contexts: ['image']
            });
            
            // Open popup for multiple operations
            chrome.contextMenus.create({
                id: 'open-popup',
                parentId: 'changeimageto-main',
                title: 'More Options...',
                contexts: ['image']
            });
        });
    }
    
    // Handle context menu clicks
    chrome.contextMenus.onClicked.addListener(async (info, tab) => {
        console.log('Context menu clicked:', info.menuItemId);
        const tabId = (tab && tab.id != null) ? tab.id : await getActiveTabId();
        const ensure = (op) => processImage(tabId, info.srcUrl, op);
        
        switch (info.menuItemId) {
            case 'remove-background':
                ensure('remove-background');
                break;
            case 'blur-background':
                ensure('blur-background');
                break;
            case 'enhance-image':
                ensure('enhance-image');
                break;
            case 'upscale-image':
                ensure('upscale-image');
                break;
            case 'open-popup':
                openPopup(tabId);
                break;
        }
    });
    
    // Handle messages from content script and popup
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        console.log('Background received message:', request);
        
        switch (request.action) {
            case 'getPageImages': {
                const respondError = (msg) => { try { sendResponse({ success: false, error: msg, images: [] }); } catch (_) {} };
                (async () => {
                    let tabId = sender && sender.tab && sender.tab.id != null ? sender.tab.id : await getActiveTabId();
                    if (tabId == null) { respondError('No active tab'); return; }
                    try {
                        const results = await chrome.scripting.executeScript({
                            target: { tabId, allFrames: true },
                            func: (opts) => {
                                const sleep = (ms) => new Promise(r => setTimeout(r, ms));
                                const extractUrlsFromStyle = (styleVal) => {
                                    const urls = [];
                                    const re = /url\(("|')?(.*?)\1\)/g; let m;
                                    while ((m = re.exec(styleVal)) !== null) { urls.push(m[2]); }
                                    return urls;
                                };
                                const collect = () => {
                                    const results = [];
                                    // <img> and <picture>
                                    document.querySelectorAll('img').forEach(im => {
                                        const src = im.currentSrc || im.src || im.getAttribute('data-src') || '';
                                        if (!src) return;
                                        const w = im.naturalWidth || im.width || 0; const h = im.naturalHeight || im.height || 0;
                                        if (w < 50 || h < 50) return; // skip tiny/tracking
                                        results.push({ src, alt: im.alt || '', width: w, height: h });
                                    });
                                    document.querySelectorAll('picture source[srcset]').forEach(s => {
                                        const srcset = s.getAttribute('srcset') || '';
                                        const first = srcset.split(',')[0]?.trim().split(' ')[0];
                                        if (first) results.push({ src: first, alt: '', width: 0, height: 0 });
                                    });
                                    // Background images
                                    document.querySelectorAll('*').forEach(el => {
                                        const cs = getComputedStyle(el);
                                        if (!cs || !cs.backgroundImage || cs.backgroundImage === 'none') return;
                                        const urls = extractUrlsFromStyle(cs.backgroundImage);
                                        urls.forEach(u => { if (u && !u.startsWith('data:')) results.push({ src: u, alt: '', width: el.clientWidth, height: el.clientHeight }); });
                                    });
                                    // Lazy attributes
                                    document.querySelectorAll('[data-src],[data-lazy-src],[data-original]').forEach(el => {
                                        const u = el.getAttribute('data-src') || el.getAttribute('data-lazy-src') || el.getAttribute('data-original');
                                        if (u) results.push({ src: u, alt: '', width: el.clientWidth, height: el.clientHeight });
                                    });
                                    // Deduplicate by absolute URL
                                    const a = document.createElement('a');
                                    const seen = new Set();
                                    return results.filter(it => {
                                        try { a.href = it.src; const abs = a.href; if (seen.has(abs)) return false; seen.add(abs); it.src = abs; return true; } catch { return false; }
                                    });
                                };
                                return (async () => {
                                    if (opts && opts.deep) {
                                        // Auto-scroll to trigger lazy content
                                        const totalSteps = 6; const step = Math.max(200, Math.floor(window.innerHeight * 0.8));
                                        for (let i = 0; i < totalSteps; i++) { window.scrollBy(0, step); await sleep(250); }
                                        await sleep(500);
                                    }
                                    const items = collect();
                                    return items.slice(0, (opts && opts.limit) || 200);
                                })();
                            },
                            args: [ { deep: !!(request && request.deep), limit: 200 } ]
                        });
                        // Aggregate from frames
                        const images = (results || []).flatMap(r => (r && r.result) ? r.result : []);
                        try { sendResponse({ success: true, images }); } catch (_) {}
                    } catch (e) {
                        respondError(e?.message || String(e));
                    }
                })();
                return true;
            }
            case 'processImage':
                const runWithTab = (tabId) => {
                    if (tabId == null) {
                        try { sendResponse({ success: false, error: 'No active tab id' }); } catch (_) {}
                        return;
                    }
                    processImage(tabId, request.imageSrc, request.operation, request.bgColor)
                        .then(result => sendResponse({ success: true, result }))
                        .catch(error => {
                            try {
                                sendResponse({ success: false, error: error?.message || String(error) });
                            } catch (_) {}
                        });
                };
                if (sender && sender.tab && sender.tab.id != null) {
                    runWithTab(sender.tab.id);
                } else {
                    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
                        runWithTab(tabs && tabs[0] ? tabs[0].id : null);
                    });
                }
                return true; // Keep message channel open
                
            case 'openPopup':
                openPopup(sender.tab.id);
                sendResponse({ success: true });
                break;
                
            case 'getSelectedImages':
                // Forward to content script
                chrome.tabs.sendMessage(sender.tab.id, request, sendResponse);
                return true;
                
            case 'clearSelection':
                // Forward to content script
                chrome.tabs.sendMessage(sender.tab.id, request, sendResponse);
                return true;
        }
    });
    
    // Helper: get active tab id
    function getActiveTabId() {
        return new Promise((resolve) => {
            try {
                chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
                    resolve(tabs && tabs[0] ? tabs[0].id : null);
                });
            } catch (_) {
                resolve(null);
            }
        });
    }

    // Process image with backend API
    async function processImage(tabId, imageSrc, operation, bgColor) {
        try {
            console.log(`Processing image: ${imageSrc} with operation: ${operation}`);
            
            // Ensure we have a valid tab id for UI updates
            if (tabId == null) {
                tabId = await getActiveTabId();
            }
            // Show loading notification (guard tabId)
            try {
                if (tabId != null) {
                    chrome.action.setBadgeText({ text: '...', tabId });
                    chrome.action.setBadgeBackgroundColor({ color: '#4CAF50', tabId });
                } else {
                    chrome.action.setBadgeText({ text: '...' });
                    chrome.action.setBadgeBackgroundColor({ color: '#4CAF50' });
                }
            } catch (_) {}
            
            // Notify popup that processing started (if open)
            try {
                chrome.runtime.sendMessage({ action: 'processingStarted', operation });
            } catch (_) {}

            // Fetch image
            const response = await fetch(imageSrc);
            const blob = await response.blob();
            
            // Create form data with explicit image content-type to satisfy backend validation
            const formData = new FormData();
            const imageType = blob.type && blob.type.startsWith('image/') ? blob.type : 'image/png';
            const file = new File([blob], 'image.png', { type: imageType });
            formData.append('file', file, 'image.png');
            
            // Determine API endpoint
            let endpoint = '';
            switch (operation) {
                case 'remove-background':
                    endpoint = '/api/remove-bg';
                    break;
                case 'change-bg-color':
                    endpoint = '/api/remove-bg';
                    break;
                case 'blur-background':
                    endpoint = '/api/blur-background';
                    break;
                case 'enhance-image':
                    endpoint = '/api/enhance-image';
                    break;
                case 'upscale-image':
                    endpoint = '/api/upscale-image';
                    break;
                default:
                    throw new Error('Unknown operation: ' + operation);
            }
            
            // Send to backend
            // Pass background color for change-bg-color
            if (operation === 'change-bg-color' && bgColor) {
                formData.append('bg_color', bgColor);
            }

            const apiResponse = await fetch(BACKEND_URL + endpoint, { method: 'POST', body: formData });
            
            if (!apiResponse.ok) {
                throw new Error(`API request failed: ${apiResponse.status}`);
            }
            
            // Get processed image and convert to data URL (MV3 service worker safe)
            const processedBlob = await apiResponse.blob();
            const processedUrl = await (async () => {
                return await new Promise((resolve, reject) => {
                    try {
                        const reader = new FileReader();
                        reader.onloadend = () => resolve(reader.result);
                        reader.onerror = reject;
                        reader.readAsDataURL(processedBlob);
                    } catch (e) {
                        reject(e);
                    }
                });
            })();
            
            // Clear badge
            try {
                if (tabId != null) {
                    chrome.action.setBadgeText({ text: '', tabId });
                } else {
                    chrome.action.setBadgeText({ text: '' });
                }
            } catch (_) {}
            
            // Send result to content script (guard tabId)
            try {
                if (tabId != null) {
                    chrome.tabs.sendMessage(tabId, {
                        action: 'showProcessedImage',
                        originalSrc: imageSrc,
                        processedUrl: processedUrl,
                        operation: operation
                    }, () => { void chrome.runtime?.lastError; });
                }
            } catch (_) {}

            // Also notify popup (if open) to show preview
            try {
                chrome.runtime.sendMessage({
                    action: 'showProcessedImage',
                    processedUrl: processedUrl,
                    operation: operation
                }, () => { void chrome.runtime.lastError; });
            } catch (_) {}

            // Persist home quick preview so popup opened later shows it on Home
            try {
                chrome.storage.local.set({ homePreviewUrl: processedUrl, homePreviewVisible: true });
            } catch (_) {}

            // Notify popup that processing ended
            try { chrome.runtime.sendMessage({ action: 'processingEnded', operation }); } catch (_) {}
            
            return {
                processedUrl: processedUrl,
                operation: operation
            };
            
        } catch (error) {
            console.error('Error processing image:', error);
            
            // Show error badge
            try {
                if (tabId != null) {
                    chrome.action.setBadgeText({ text: '!', tabId });
                    chrome.action.setBadgeBackgroundColor({ color: '#f44336', tabId });
                } else {
                    chrome.action.setBadgeText({ text: '!' });
                    chrome.action.setBadgeBackgroundColor({ color: '#f44336' });
                }
            } catch (_) {}
            
            // Send error to content script
            chrome.tabs.sendMessage(tabId, {
                action: 'showError',
                error: error.message
            });
            // Notify popup that processing ended (with error)
            try { chrome.runtime.sendMessage({ action: 'processingEnded', operation }); } catch (_) {}
            
            throw error;
        }
    }
    
    // Open popup
    function openPopup(tabId) {
        chrome.action.openPopup();
    }
    
    // Handle extension icon click
    chrome.action.onClicked.addListener((tab) => {
        chrome.action.openPopup();
    });
    
    // Clean up on startup
    chrome.runtime.onStartup.addListener(() => {
        console.log('ChangeImageTo extension started');
        setupContextMenus();
    });
    
})();
