// Popup script for ChangeImageTo Chrome Extension
// Handles popup UI interactions and communication with content script

(function() {
    'use strict';
    
    // DOM elements
    const statusElement = document.getElementById('status');
    const selectedImagesSection = document.getElementById('selectedImagesSection');
    const selectedImagesList = document.getElementById('selectedImagesList');
    const clearSelectionBtn = document.getElementById('clearSelection');
    const actionBtns = document.querySelectorAll('.action-btn');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const resultModal = document.getElementById('resultModal');
    const resultImage = document.getElementById('resultImage');
    const downloadResultBtn = document.getElementById('downloadResult');
    const closeModalBtn = document.getElementById('closeModal');
    const modalClose = document.getElementById('modalClose');
    const openOptionsBtn = document.getElementById('openOptions');
    
    // State
    let selectedImages = [];
    let currentTab = null;
    
    // Initialize popup
    async function init() {
        console.log('Popup initialized');
        
        // Get current tab
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        currentTab = tabs[0];
        
        // Setup event listeners
        setupEventListeners();
        
        // Load selected images
        await loadSelectedImages();
        
        // Update status
        updateStatus('Ready');
    }
    
    // Setup event listeners
    function setupEventListeners() {
        // Clear selection button
        clearSelectionBtn.addEventListener('click', clearSelection);
        
        // Action buttons
        actionBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const operation = e.currentTarget.dataset.operation;
                handleQuickAction(operation);
            });
        });
        
        // Modal controls
        closeModalBtn.addEventListener('click', closeModal);
        modalClose.addEventListener('click', closeModal);
        downloadResultBtn.addEventListener('click', downloadResult);
        
        // Options button
        openOptionsBtn.addEventListener('click', () => {
            chrome.runtime.openOptionsPage();
        });
        
        // Listen for messages from background script
        chrome.runtime.onMessage.addListener(handleMessage);
    }
    
    // Handle messages from background script
    function handleMessage(request, sender, sendResponse) {
        switch (request.action) {
            case 'showProcessedImage':
                showProcessedImage(request);
                break;
            case 'showError':
                showError(request.error);
                break;
        }
    }
    
    // Load selected images from content script
    async function loadSelectedImages() {
        try {
            const response = await chrome.tabs.sendMessage(currentTab.id, { action: 'getSelectedImages' });
            selectedImages = response.images || [];
            
            if (selectedImages.length > 0) {
                showSelectedImages();
            } else {
                hideSelectedImages();
            }
        } catch (error) {
            console.log('No content script or no selected images:', error);
            hideSelectedImages();
        }
    }
    
    // Show selected images
    function showSelectedImages() {
        selectedImagesSection.style.display = 'block';
        selectedImagesList.innerHTML = '';
        
        selectedImages.forEach((img, index) => {
            const item = document.createElement('div');
            item.className = 'selected-image-item';
            item.innerHTML = `
                <img src="${img.src}" alt="${img.alt}" class="selected-image-preview">
                <div class="selected-image-info">
                    <div class="selected-image-name">${img.alt}</div>
                    <div class="selected-image-size">${img.width} × ${img.height}</div>
                </div>
            `;
            selectedImagesList.appendChild(item);
        });
    }
    
    // Hide selected images section
    function hideSelectedImages() {
        selectedImagesSection.style.display = 'none';
    }
    
    // Clear selection
    async function clearSelection() {
        try {
            await chrome.tabs.sendMessage(currentTab.id, { action: 'clearSelection' });
            selectedImages = [];
            hideSelectedImages();
            updateStatus('Selection cleared');
        } catch (error) {
            console.error('Error clearing selection:', error);
        }
    }
    
    // Handle quick action
    async function handleQuickAction(operation) {
        if (selectedImages.length === 0) {
            showError('Please select images first by clicking on them on the webpage');
            return;
        }
        
        // For now, process the first selected image
        const image = selectedImages[0];
        await processImage(image.src, operation);
    }
    
    // Process image
    async function processImage(imageSrc, operation) {
        try {
            showLoading(true);
            updateStatus(`Processing ${operation}...`);
            
            // Send message to background script
            const response = await chrome.runtime.sendMessage({
                action: 'processImage',
                imageSrc: imageSrc,
                operation: operation
            });
            
            if (response.success) {
                showProcessedImage({
                    processedUrl: response.result.processedUrl,
                    operation: operation
                });
            } else {
                showError(response.error || 'Failed to process image');
            }
        } catch (error) {
            console.error('Error processing image:', error);
            showError(error.message || 'Failed to process image');
        } finally {
            showLoading(false);
        }
    }
    
    // Show processed image in modal
    function showProcessedImage(data) {
        resultImage.src = data.processedUrl;
        document.getElementById('modalTitle').textContent = `Image ${data.operation.replace('-', ' ')} completed`;
        resultModal.style.display = 'flex';
        updateStatus('Processing complete');
        
        // Store the processed URL for download
        downloadResultBtn.dataset.downloadUrl = data.processedUrl;
    }
    
    // Download result
    function downloadResult() {
        const downloadUrl = downloadResultBtn.dataset.downloadUrl;
        if (downloadUrl) {
            chrome.downloads.download({
                url: downloadUrl,
                filename: `changeimageto-processed-${Date.now()}.png`
            });
            closeModal();
        }
    }
    
    // Close modal
    function closeModal() {
        resultModal.style.display = 'none';
        if (resultImage.src) {
            URL.revokeObjectURL(resultImage.src);
            resultImage.src = '';
        }
    }
    
    // Show loading overlay
    function showLoading(show) {
        loadingOverlay.style.display = show ? 'flex' : 'none';
    }
    
    // Show error
    function showError(message) {
        updateStatus(`Error: ${message}`);
        // You could also show a toast notification here
        console.error(message);
    }
    
    // Update status
    function updateStatus(message) {
        statusElement.textContent = message;
        
        // Auto-clear status after 3 seconds
        setTimeout(() => {
            if (statusElement.textContent === message) {
                statusElement.textContent = 'Ready';
            }
        }, 3000);
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
})();
