// Content script for ChangeImageTo Chrome Extension
// This script runs on every webpage to detect images and add context menu options

(function() {
    'use strict';
    
    // Configuration
    const BACKEND_URL = 'https://bgremover-backend-121350814881.us-central1.run.app';
    
    // Track selected images
    let selectedImages = new Set();
    
    // Safe check for extension context
    function isExtensionAlive() {
        try { return !!(chrome && chrome.runtime && chrome.runtime.id); } catch { return false; }
    }

    // Initialize the content script
    function init() {
        console.log('ChangeImageTo extension loaded');
        
        // Add visual indicators to images
        addImageIndicators();
        
        // Listen for messages from background script (guarded)
        try { if (isExtensionAlive()) chrome.runtime.onMessage.addListener(handleMessage); } catch {}
        
        // Listen for image clicks
        // Use capture phase so we get the click before site handlers navigate
        document.addEventListener('click', handleImageClick, true);
        
        // Listen for context menu events
        document.addEventListener('contextmenu', handleContextMenu);

        // Track dynamic DOM changes and re-tag late-loaded images
        try {
            const observer = new MutationObserver(() => addImageIndicators());
            observer.observe(document.documentElement, { childList: true, subtree: true });
        } catch {}
    }
    
    // Add visual indicators to images
    function addImageIndicators() {
        const images = document.querySelectorAll('img');
        
        images.forEach(img => {
            // Skip if already processed
            if (img.dataset.changeimagetoProcessed) return;
            
            // Add data attribute to mark as processed
            img.dataset.changeimagetoProcessed = 'true';
            
            // Add hover effect
            img.style.transition = 'opacity 0.2s ease';
            img.addEventListener('mouseenter', () => {
                if (!img.classList.contains('changeimageto-selected')) {
                    img.style.opacity = '0.8';
                }
            });
            
            img.addEventListener('mouseleave', () => {
                if (!img.classList.contains('changeimageto-selected')) {
                    img.style.opacity = '1';
                }
            });
        });
    }
    
    // Handle image clicks
    function handleImageClick(event) {
        if (event.target && event.target.tagName === 'IMG') {
            // Require a modifier to avoid breaking normal site clicks
            const withModifier = event.altKey || event.metaKey || event.ctrlKey;
            if (!withModifier) return; // only select on Alt/⌘/Ctrl click

            const img = event.target;
            
            // Toggle selection
            if (img.classList.contains('changeimageto-selected')) {
                img.classList.remove('changeimageto-selected');
                selectedImages.delete(img);
            } else {
                img.classList.add('changeimageto-selected');
                selectedImages.add(img);
            }
            
            // Prevent site navigation for modified selection clicks
            event.preventDefault();
            event.stopPropagation();

            // Update selection count
            updateSelectionCount();
        }
    }
    
    // Handle context menu events
    function handleContextMenu(event) {
        if (event.target.tagName === 'IMG') {
            const img = event.target;
            
            // Store the clicked image for context menu
            try {
                if (!isExtensionAlive()) return;
                chrome.storage?.local?.set({ 
                    contextImage: {
                        src: img.src,
                        alt: img.alt || 'Image',
                        width: img.naturalWidth || img.width,
                        height: img.naturalHeight || img.height
                    }
                }, () => { void chrome.runtime?.lastError; });
            } catch {}
        }
    }
    
    // Handle messages from background script
    function handleMessage(request, sender, sendResponse) {
        switch (request.action) {
            case 'getSelectedImages':
                const images = Array.from(selectedImages).map(img => ({
                    src: img.src,
                    alt: img.alt || 'Image',
                    width: img.naturalWidth || img.width,
                    height: img.naturalHeight || img.height
                }));
                sendResponse({ images });
                break;
                
            case 'clearSelection':
                clearSelection();
                sendResponse({ success: true });
                break;
            default:
                // no-op for other actions
                break;
        }
    }
    
    // Update selection count indicator
    function updateSelectionCount() {
        // Remove existing indicator
        const existingIndicator = document.getElementById('changeimageto-selection-count');
        if (existingIndicator) {
            existingIndicator.remove();
        }
        
        // Add new indicator if images are selected
        if (selectedImages.size > 0) {
            const indicator = document.createElement('div');
            indicator.id = 'changeimageto-selection-count';
            indicator.innerHTML = `
                <div style="
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #4CAF50;
                    color: white;
                    padding: 10px 15px;
                    border-radius: 25px;
                    font-family: Arial, sans-serif;
                    font-size: 14px;
                    font-weight: bold;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                    z-index: 10000;
                    cursor: pointer;
                ">
                    ${selectedImages.size} image${selectedImages.size > 1 ? 's' : ''} selected
                    <br><small>Click to edit</small>
                </div>
            `;
            
            indicator.addEventListener('click', () => {
                try {
                    if (chrome && chrome.runtime && chrome.runtime.id) {
                        chrome.runtime.sendMessage({ action: 'openPopup' }, () => {
                            // ignore lastError if service worker was sleeping; Chrome wakes it up
                            void chrome.runtime.lastError;
                        });
                    }
                } catch (_) {
                    // Silently ignore if extension context is invalidated; user can click toolbar icon
                }
            });
            
            document.body.appendChild(indicator);
        }
    }
    
    // Clear all selections
    function clearSelection() {
        selectedImages.forEach(img => {
            img.classList.remove('changeimageto-selected');
        });
        selectedImages.clear();
        
        const indicator = document.getElementById('changeimageto-selection-count');
        if (indicator) {
            indicator.remove();
        }
    }
    
    // No processing here; background handles API calls
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
})();
