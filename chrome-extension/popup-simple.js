// Popup script for ChangeImageTo Chrome Extension - Simple Version
// Handles popup UI interactions and communication with content script

let selectedImages = [];
let currentTab = null;
let lastProcessedDataUrl = null;
let pageImages = [];
let pageIndex = 0;
const PAGE_SIZE = 50;
let __loadingStartedAt = 0;
let pickedResults = [];
let __batchProcessing = false;
let __batchTargetCount = 0;

// Persistence helpers
async function saveUIState() {
	try {
		const gallery = document.getElementById('pickedResultsGallery');
		const grid = document.getElementById('pickedResultsGrid');
		const pickedPreviewSection = document.getElementById('pickedResultSection');
		const pickedPreviewImg = document.getElementById('pickedResultPreview');
		const uploadedSection = document.getElementById('resultSection');
		const uploadedImg = document.getElementById('resultPreview');
		const homeSection = document.getElementById('homeQuickPreviewSection');
		const homeImg = document.getElementById('homeQuickPreview');
		await chrome.storage.local.set({
			pickedResults: Array.isArray(pickedResults) ? pickedResults : [],
			pickedGalleryVisible: !!(gallery && gallery.style.display === 'block' && grid && grid.children.length > 0),
			pickedPreviewVisible: !!(pickedPreviewSection && pickedPreviewSection.style.display === 'block'),
			pickedPreviewUrl: (pickedPreviewImg && pickedPreviewImg.src) || '',
			uploadedPreviewVisible: !!(uploadedSection && uploadedSection.style.display === 'block'),
			uploadedPreviewUrl: (uploadedImg && uploadedImg.src) || '',
			homePreviewVisible: !!(homeSection && homeSection.style.display === 'block'),
			homePreviewUrl: (homeImg && homeImg.src) || ''
		});
	} catch (_) {}
}

async function restoreUIState() {
	try {
    const data = await chrome.storage.local.get(['pickedResults','pickedGalleryVisible','pickedPreviewVisible','pickedPreviewUrl','uploadedPreviewVisible','uploadedPreviewUrl','homePreviewVisible','homePreviewUrl']);
		// Restore picked gallery
		const gallery = document.getElementById('pickedResultsGallery');
		const grid = document.getElementById('pickedResultsGrid');
		const prog = document.getElementById('pickedProgress');
		if (Array.isArray(data.pickedResults) && data.pickedResults.length) {
			pickedResults = data.pickedResults.slice();
			if (gallery && grid) {
				gallery.style.display = data.pickedGalleryVisible ? 'block' : 'none';
				grid.innerHTML = '';
				pickedResults.forEach((url) => {
					const item = document.createElement('div');
					item.style.border = '1px solid #eee';
					item.style.borderRadius = '4px';
					item.style.background = '#fff';
					item.style.padding = '4px';
					const img = document.createElement('img');
					img.src = url;
					img.style.width = '100%';
					img.style.height = '80px';
					img.style.objectFit = 'cover';
					const btn = document.createElement('button');
					btn.className = 'btn';
					btn.style.margin = '6px 0 0 0';
					btn.textContent = 'Download';
					btn.addEventListener('click', () => {
						const a = document.createElement('a');
						a.href = url; a.download = `changeimageto-processed-${Date.now()}.png`; a.click();
					});
					item.appendChild(img);
					item.appendChild(btn);
					grid.appendChild(item);
				});
				if (prog) prog.textContent = `${pickedResults.length}/${pickedResults.length}`;
			}
		}
		// Restore picked preview
		const pickedPreviewSection = document.getElementById('pickedResultSection');
		const pickedPreviewImg = document.getElementById('pickedResultPreview');
		if (pickedPreviewSection && pickedPreviewImg) {
			pickedPreviewImg.src = data.pickedPreviewUrl || '';
			pickedPreviewSection.style.display = data.pickedPreviewVisible ? 'block' : 'none';
		}
		// Restore uploaded preview
		const uploadedSection = document.getElementById('resultSection');
		const uploadedImg = document.getElementById('resultPreview');
		if (uploadedSection && uploadedImg) {
			uploadedImg.src = data.uploadedPreviewUrl || '';
			uploadedSection.style.display = data.uploadedPreviewVisible ? 'block' : 'none';
			ensureUploadCTA();
		}
		// Restore home quick preview
		const homeSection = document.getElementById('homeQuickPreviewSection');
		const homeImg = document.getElementById('homeQuickPreview');
		if (homeSection && homeImg) {
			homeImg.src = data.homePreviewUrl || '';
			homeSection.style.display = data.homePreviewVisible ? 'block' : 'none';
		}
	} catch (_) {}
}

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
    
    updateStatus('Ready to edit images');
    
    // Wire preview buttons
    document.getElementById('downloadBtn').addEventListener('click', () => {
        if (!lastProcessedDataUrl) return;
        const a = document.createElement('a');
        a.href = lastProcessedDataUrl;
        a.download = `changeimageto-processed-${Date.now()}.png`;
        document.body.appendChild(a);
        a.click();
        a.remove();
    });
    // Home quick preview buttons
    const hdl = document.getElementById('homeQuickDownloadBtn');
    if (hdl) hdl.addEventListener('click', () => {
        if (!lastProcessedDataUrl) return; const a = document.createElement('a'); a.href = lastProcessedDataUrl; a.download = `changeimageto-processed-${Date.now()}.png`; a.click();
    });
    const hclr = document.getElementById('homeQuickClearBtn');
    if (hclr) hclr.addEventListener('click', () => {
        lastProcessedDataUrl = null; const img = document.getElementById('homeQuickPreview'); if (img) img.src = ''; const sec = document.getElementById('homeQuickPreviewSection'); if (sec) sec.style.display = 'none'; saveUIState();
    });
    document.getElementById('clearPreviewBtn').addEventListener('click', () => {
        lastProcessedDataUrl = null;
        document.getElementById('resultPreview').src = '';
        document.getElementById('resultSection').style.display = 'none';
        ensureUploadCTA();
        // Reset file input so selecting the same file again will fire 'change'
        const up = document.getElementById('uploadInput');
        if (up) up.value = '';
        saveUIState();
    });

    // Tabs
    const viewHome = document.getElementById('viewHome');
    const viewPicked = document.getElementById('viewPicked');
    const viewUploaded = document.getElementById('viewUploaded');
    const tabHome = document.getElementById('tabHome');
    const tabPicked = document.getElementById('tabPicked');
    const tabUploaded = document.getElementById('tabUploaded');

    // Move picker/preview sections to their tab views so Home stays clean
    try {
        const picker = document.getElementById('pageImagesSection');
        const actions = document.getElementById('actionsToolbar');
        const preview = document.getElementById('resultSection');
        if (picker && viewPicked) viewPicked.appendChild(picker);
        if (actions && viewPicked) viewPicked.appendChild(actions);
        if (preview && viewUploaded) {
            const uplToolbar = document.getElementById('uploadedActionsToolbar');
            if (uplToolbar) {
                viewUploaded.insertBefore(preview, uplToolbar);
            } else {
                viewUploaded.appendChild(preview);
            }
        }
    } catch (_) {}

    window.switchTab = function(tab) {
        viewHome.style.display = tab === 'home' ? 'block' : 'none';
        viewPicked.style.display = tab === 'picked' ? 'block' : 'none';
        viewUploaded.style.display = tab === 'uploaded' ? 'block' : 'none';
        const ACTIVE = '#0b0f13';
        const INACTIVE = '#6c757d';
        tabHome.style.background = tab === 'home' ? ACTIVE : INACTIVE;
        tabPicked.style.background = tab === 'picked' ? ACTIVE : INACTIVE;
        tabUploaded.style.background = tab === 'uploaded' ? ACTIVE : INACTIVE;
    };
    tabHome.addEventListener('click', () => window.switchTab('home'));
    tabPicked.addEventListener('click', () => { ensurePickedCTA(); window.switchTab('picked'); });
    tabUploaded.addEventListener('click', () => { ensureUploadCTA(); window.switchTab('uploaded'); });

    // Picked tab preview controls
    const pickedResultSection = document.getElementById('pickedResultSection');
    const pickedResultPreview = document.getElementById('pickedResultPreview');
    const pickedDownloadBtn = document.getElementById('pickedDownloadBtn');
    const pickedClearBtn = document.getElementById('pickedClearBtn');
    if (pickedDownloadBtn) {
        pickedDownloadBtn.addEventListener('click', () => {
            if (!lastProcessedDataUrl) return;
            const a = document.createElement('a');
            a.href = lastProcessedDataUrl;
            a.download = `changeimageto-processed-${Date.now()}.png`;
            document.body.appendChild(a);
            a.click();
            a.remove();
        });
    }
    if (pickedClearBtn) {
        pickedClearBtn.addEventListener('click', () => {
            lastProcessedDataUrl = null;
            if (pickedResultPreview) pickedResultPreview.src = '';
            if (pickedResultSection) pickedResultSection.style.display = 'none';
            ensurePickedCTA();
            saveUIState();
        });
    }

    // Restore any persisted UI state after layout is ready
    await restoreUIState();
}

// Setup event listeners
function setupEventListeners() {
    document.getElementById('removeBackgroundBtn').addEventListener('click', () => {
        processImage('remove-background');
    });
    
    document.getElementById('blurBackgroundBtn').addEventListener('click', () => {
        processImage('blur-background');
    });
    
    document.getElementById('enhanceImageBtn').addEventListener('click', () => {
        processImage('enhance-image');
    });
    
    document.getElementById('upscaleImageBtn').addEventListener('click', () => {
        processImage('upscale-image');
    });
    // Change background color via color picker
    const changeBgBtn = document.getElementById('changeBgColorBtn');
    if (changeBgBtn) changeBgBtn.addEventListener('click', async () => {
        await pickAndApplyBgColorForSelected();
    });
    
    document.getElementById('refreshBtn').addEventListener('click', () => {
        loadSelectedImages();
    });
    
    // Upload and page images
    document.getElementById('uploadBtn').addEventListener('click', () => {
        document.getElementById('uploadInput').click();
    });
    document.getElementById('uploadInput').addEventListener('change', async (e) => {
        const file = e.target.files && e.target.files[0];
        if (!file) return;
        // Do not process automatically; just show preview in Uploaded tab and enable actions
        showLoading('Loading preview...');
        const dataUrl = await fileToDataURL(file);
        lastProcessedDataUrl = null;
        const preview = document.getElementById('resultPreview');
        preview.src = dataUrl;
        document.getElementById('resultSection').style.display = 'block';
        const uplToolbar = document.getElementById('uploadedActionsToolbar');
        if (uplToolbar) uplToolbar.style.display = 'block';
        const empty = document.getElementById('emptyUploaded');
        if (empty) empty.style.display = 'none';
        window.switchTab('uploaded');
        // Wire uploaded actions to process this file on demand
        wireUploadedActions(file);
        hideLoading();
        // Re-evaluate empty/toolbar visibility
        ensureUploadCTA();
        saveUIState();
    });
    const pickFromPageBtn = document.getElementById('pickFromPageBtn');
    if (pickFromPageBtn) {
        pickFromPageBtn.addEventListener('click', async () => {
        try {
            updateStatus('Scanning page for images...');
            showLoading('Scanning page for images...');
            const deep = document.getElementById('deepScanToggle')?.checked || false; // if visible
            const resp = await chrome.runtime.sendMessage({ action: 'getPageImages', deep });
            hideLoading();
            if (resp && resp.success) {
                pageImages = resp.images || [];
                pageIndex = 0;
                renderPageImages(paginate(pageImages));
                updateActionsToolbar();
                if (window.switchTab) {
                    window.switchTab('picked');
                } else {
                    // Fallback if switchTab not yet defined
                    const viewHome = document.getElementById('viewHome');
                    const viewPicked = document.getElementById('viewPicked');
                    const viewUploaded = document.getElementById('viewUploaded');
                    const tabHome = document.getElementById('tabHome');
                    const tabPicked = document.getElementById('tabPicked');
                    const tabUploaded = document.getElementById('tabUploaded');
                    if (viewHome && viewPicked && viewUploaded && tabHome && tabPicked && tabUploaded) {
                        viewHome.style.display = 'none';
                        viewPicked.style.display = 'block';
                        viewUploaded.style.display = 'none';
                        tabHome.style.background = '#6c757d';
                        tabPicked.style.background = '#0b0f13';
                        tabUploaded.style.background = '#6c757d';
                    }
                }
                ensurePickedCTA();
                updateStatus(`Found ${pageImages.length} image(s)`);
            } else {
                pageImages = [];
                renderPageImages([]);
                const errorMsg = resp && resp.error ? resp.error : 'Failed to get page images';
                updateStatus(`Error: ${errorMsg}`);
            }
        } catch (e) {
            hideLoading();
            console.error('Error picking images from page:', e);
            updateStatus(`Error: ${e.message || e}`);
            pageImages = [];
            renderPageImages([]);
        }
    });
    } else {
        console.error('pickFromPageBtn not found in DOM');
    }

    // Empty state CTAs
    const emptyPickBtn = document.getElementById('ctaPickFromPage');
    if (emptyPickBtn) emptyPickBtn.addEventListener('click', () => document.getElementById('pickFromPageBtn').click());
    const emptyUploadBtn = document.getElementById('ctaUploadFromEmpty');
    if (emptyUploadBtn) emptyUploadBtn.addEventListener('click', () => document.getElementById('uploadInput').click());

    document.getElementById('selectAllBtn').addEventListener('click', () => toggleAll(true));
    document.getElementById('clearAllBtn').addEventListener('click', () => toggleAll(false));
    document.getElementById('showMoreBtn').addEventListener('click', () => {
        if (!pageImages.length) return;
        const grid = document.getElementById('pageImagesGrid');
        const more = paginate(pageImages);
        appendPageImages(more, grid);
    });
    
    // Review link handler
    const reviewLink = document.getElementById('reviewLink');
    if (reviewLink) {
        reviewLink.addEventListener('click', (e) => {
            e.preventDefault();
            // Get extension ID and open Chrome Web Store review page
            const extensionId = chrome.runtime.id;
            const reviewUrl = `https://chrome.google.com/webstore/detail/${extensionId}/reviews`;
            chrome.tabs.create({ url: reviewUrl });
        });
    }
}

// Load selected images from content script
async function loadSelectedImages() {
    try {
        const response = await chrome.tabs.sendMessage(currentTab.id, { action: 'getSelectedImages' });
        selectedImages = response.images || [];
        
        if (selectedImages.length > 0) {
            updateStatus(`${selectedImages.length} image(s) selected`);
        } else {
            updateStatus('No images selected - right-click on images to select them');
        }
    } catch (error) {
        console.log('No content script or no selected images:', error);
        updateStatus('Right-click on images to select them');
    }
}

// Process image with backend
async function processImage(operation) {
    if (selectedImages.length === 0) {
        updateStatus('Please select images first by clicking on them on the webpage');
        return;
    }
    
    try {
        showLoading(`Processing ${operation}...`);
        
        // For now, process the first selected image
        const image = selectedImages[0];
        
        // Send message to background script
        const response = await chrome.runtime.sendMessage({
            action: 'processImage',
            imageSrc: image.src,
            operation: operation
        });
        
        if (response && response.success) {
            updateStatus(`${operation} completed successfully!`);
            
            // Show preview and enable manual download
            lastProcessedDataUrl = response.result.processedUrl;
            const imgEl = document.getElementById('resultPreview');
            imgEl.src = lastProcessedDataUrl;
            document.getElementById('resultSection').style.display = 'block';
        } else {
            updateStatus(`Error: ${response.error || 'Failed to process image'}`);
        }
    } catch (error) {
        console.error('Error processing image:', error);
        updateStatus(`Error: ${error.message || 'Failed to process image'}`);
    } finally { hideLoading(); }
}

function updateStatus(message) {
    document.getElementById('status').textContent = message;
}

// Process a locally uploaded file directly
async function processFileDirect(file, operation) {
    try {
        showLoading(`Processing ${operation}...`);
        const formData = new FormData();
        formData.append('file', file, file.name || 'image.png');
        let endpoint = '';
        switch (operation) {
            case 'remove-background': endpoint = '/api/remove-bg'; break;
            case 'blur-background': endpoint = '/api/blur-background'; break;
            case 'enhance-image': endpoint = '/api/enhance-image'; break;
            case 'upscale-image': endpoint = '/api/upscale-image'; break;
            default: endpoint = '/api/remove-bg';
        }
        const apiResponse = await fetch('https://bgremover-backend-121350814881.us-central1.run.app' + endpoint, {
            method: 'POST',
            body: formData
        });
        if (!apiResponse.ok) throw new Error(`API request failed: ${apiResponse.status}`);
        const blob = await apiResponse.blob();
        const dataUrl = await new Promise((resolve, reject) => {
            const r = new FileReader();
            r.onloadend = () => resolve(r.result);
            r.onerror = reject;
            r.readAsDataURL(blob);
        });
        lastProcessedDataUrl = dataUrl;
        document.getElementById('resultPreview').src = dataUrl;
        document.getElementById('resultSection').style.display = 'block';
        updateStatus('Processing complete');
    } catch (e) {
        updateStatus(`Error: ${e.message || e}`);
    } finally { hideLoading(); }
}

// Ask user for a color and apply to selected/picked image(s)
async function pickAndApplyBgColorForSelected() {
    try {
        // Prefer picked selections if in picked tab
        const inPicked = document.getElementById('viewPicked')?.style.display === 'block';
        const pickedTargets = getSelectedPageImages();
        const hasPicked = inPicked && pickedTargets.length > 0;
        const singleSelectedSrc = (selectedImages[0] && selectedImages[0].src) || null;

        // Show color picker
        const input = document.createElement('input');
        input.type = 'color';
        input.value = '#ffffff';
        const color = await new Promise((resolve) => {
            input.addEventListener('input', () => resolve(input.value), { once: true });
            input.click();
        });
        if (!color) return;

        if (hasPicked) {
            await runBatchChangeBgColor(color);
            return;
        }

        if (!singleSelectedSrc) {
            updateStatus('Please select or upload an image first');
            return;
        }

        showLoading('Applying background color...');
        const resp = await chrome.runtime.sendMessage({ action: 'processImage', imageSrc: singleSelectedSrc, operation: 'change-bg-color', bgColor: color });
        if (resp && resp.success) {
            lastProcessedDataUrl = resp.result.processedUrl;
            const imgEl = document.getElementById('resultPreview');
            if (imgEl) {
                imgEl.src = lastProcessedDataUrl;
                const section = document.getElementById('resultSection');
                if (section) section.style.display = 'block';
                await saveUIState();
            }
        } else {
            updateStatus(`Error: ${resp?.error || 'Failed to change background color'}`);
        }
    } catch (e) {
        updateStatus(`Error: ${e.message || e}`);
    } finally { hideLoading(); }
}

async function runBatchChangeBgColor(color) {
    const targets = getSelectedPageImages();
    if (!targets.length) { updateStatus('Select images first'); return; }
    showLoading(`Applying color to ${targets.length} image(s)...`);
    const gallery = document.getElementById('pickedResultsGallery');
    const grid = document.getElementById('pickedResultsGrid');
    const prog = document.getElementById('pickedProgress');
    if (gallery && grid) { gallery.style.display = 'block'; grid.innerHTML = ''; }
    pickedResults = [];
    __batchProcessing = targets.length > 1;
    __batchTargetCount = targets.length;
    const sec = document.getElementById('pickedResultSection');
    if (sec) sec.style.display = 'none';
    const prv = document.getElementById('pickedResultPreview');
    if (prv) prv.src = '';
    await saveUIState();
    for (const src of targets) {
        const resp = await chrome.runtime.sendMessage({ action: 'processImage', imageSrc: src, operation: 'change-bg-color', bgColor: color });
        if (resp && resp.success) {
            lastProcessedDataUrl = resp.result.processedUrl;
            if (grid) {
                const item = document.createElement('div');
                item.style.border = '1px solid #eee';
                item.style.borderRadius = '4px';
                item.style.background = '#fff';
                item.style.padding = '4px';
                const img = document.createElement('img');
                img.src = lastProcessedDataUrl;
                img.style.width = '100%';
                img.style.height = '80px';
                img.style.objectFit = 'cover';
                const btn = document.createElement('button');
                btn.className = 'btn';
                btn.style.margin = '6px 0 0 0';
                btn.textContent = 'Download';
                btn.addEventListener('click', () => { const a = document.createElement('a'); a.href = lastProcessedDataUrl; a.download = `changeimageto-processed-${Date.now()}.png`; a.click(); });
                item.appendChild(img);
                item.appendChild(btn);
                grid.appendChild(item);
                pickedResults.push(lastProcessedDataUrl);
                if (prog) prog.textContent = `${pickedResults.length}/${targets.length}`;
                await saveUIState();
            }
        }
    }
    updateStatus('Processing complete');
    hideLoading();
    __batchProcessing = false; __batchTargetCount = 0;
    const finalizeSec = document.getElementById('pickedResultSection'); if (finalizeSec) finalizeSec.style.display = 'none';
    await saveUIState();
}

function fileToDataURL(file) {
    return new Promise((resolve, reject) => {
        const r = new FileReader();
        r.onloadend = () => resolve(r.result);
        r.onerror = reject;
        r.readAsDataURL(file);
    });
}

function wireUploadedActions(file) {
    const run = async (operation) => {
        try {
            updateStatus(`Processing ${operation}...`);
            const formData = new FormData();
            formData.append('file', file, file.name || 'image.png');
            let endpoint = '';
            switch (operation) {
                case 'remove-background': endpoint = '/api/remove-bg'; break;
                case 'blur-background': endpoint = '/api/blur-background'; break;
                case 'enhance-image': endpoint = '/api/enhance-image'; break;
                case 'upscale-image': endpoint = '/api/upscale-image'; break;
                default: endpoint = '/api/remove-bg';
            }
            showLoading(`Processing ${operation}...`);
            const apiResponse = await fetch('https://bgremover-backend-121350814881.us-central1.run.app' + endpoint, { method: 'POST', body: formData });
            if (!apiResponse.ok) throw new Error(`API request failed: ${apiResponse.status}`);
            const blob = await apiResponse.blob();
            const dataUrl = await new Promise((resolve, reject) => { const r = new FileReader(); r.onloadend = () => resolve(r.result); r.onerror = reject; r.readAsDataURL(blob); });
            lastProcessedDataUrl = dataUrl;
            const prv = document.getElementById('resultPreview');
            prv.src = dataUrl;
            document.getElementById('resultSection').style.display = 'block';
            updateStatus('Processing complete');
            await saveUIState();
        } catch (e) {
            updateStatus(`Error: ${e.message || e}`);
        } finally { hideLoading(); }
    };

    const set = (id, op) => {
        const el = document.getElementById(id);
        if (!el) return;
        // Remove previous listeners by cloning
        const clone = el.cloneNode(true);
        el.parentNode.replaceChild(clone, el);
        clone.addEventListener('click', () => run(op));
    };
    set('uplRemoveBg', 'remove-background');
    set('uplBlurBg', 'blur-background');
    set('uplEnhance', 'enhance-image');
    set('uplUpscale', 'upscale-image');
    // Change BG color for uploaded image
    const uplColor = document.getElementById('uplChangeBgColor');
    if (uplColor) {
        const clone = uplColor.cloneNode(true);
        uplColor.parentNode.replaceChild(clone, uplColor);
        clone.addEventListener('click', async () => {
            const input = document.createElement('input'); input.type = 'color'; input.value = '#ffffff';
            const color = await new Promise((resolve) => { input.addEventListener('input', () => resolve(input.value), { once: true }); input.click(); });
            if (!color) return;
            try {
                updateStatus('Applying background color...');
                const formData = new FormData(); formData.append('file', file, file.name || 'image.png'); formData.append('bg_color', color);
                showLoading('Processing change background color...');
                const apiResponse = await fetch('https://bgremover-backend-121350814881.us-central1.run.app' + '/api/remove-bg', { method: 'POST', body: formData });
                if (!apiResponse.ok) throw new Error(`API request failed: ${apiResponse.status}`);
                const blob = await apiResponse.blob();
                const dataUrl = await new Promise((resolve, reject) => { const r = new FileReader(); r.onloadend = () => resolve(r.result); r.onerror = reject; r.readAsDataURL(blob); });
                lastProcessedDataUrl = dataUrl; const prv = document.getElementById('resultPreview'); prv.src = dataUrl; document.getElementById('resultSection').style.display = 'block'; updateStatus('Processing complete'); await saveUIState();
            } catch (e) { updateStatus(`Error: ${e.message || e}`); }
            finally { hideLoading(); }
        });
    }
    const dl = document.getElementById('uplDownloadOriginal');
    if (dl) {
        const clone = dl.cloneNode(true);
        dl.parentNode.replaceChild(clone, dl);
        clone.addEventListener('click', () => downloadUrl(URL.createObjectURL(file)));
    }
}

function renderPageImages(images) {
    const section = document.getElementById('pageImagesSection');
    const empty = document.getElementById('emptyPicked');
    const grid = document.getElementById('pageImagesGrid');
    grid.innerHTML = '';
    if (!images || images.length === 0) {
        section.style.display = 'none';
        document.getElementById('actionsToolbar').style.display = 'none';
        if (empty) empty.style.display = 'block';
        return;
    }
    appendPageImages(images, grid);
    section.style.display = 'block';
    document.getElementById('actionsToolbar').style.display = 'block';
    if (empty) empty.style.display = 'none';
}

function appendPageImages(images, grid) {
    (images || []).forEach((img, idx) => {
        const wrapper = document.createElement('div');
        wrapper.style.position = 'relative';
        wrapper.style.border = '1px solid #eee';
        wrapper.style.borderRadius = '4px';
        wrapper.style.padding = '4px';
        wrapper.style.background = '#fff';

        const thumb = document.createElement('img');
        thumb.src = img.src;
        thumb.alt = img.alt || 'Image';
        thumb.style.width = '100%';
        thumb.style.height = '60px';
        thumb.style.objectFit = 'cover';

        const chk = document.createElement('input');
        chk.type = 'checkbox';
        chk.dataset.src = img.src;
        chk.style.position = 'absolute';
        chk.style.top = '6px';
        chk.style.left = '6px';
        chk.addEventListener('change', updateActionsToolbar);

        const dl = document.createElement('button');
        dl.textContent = '⬇️';
        dl.title = 'Download original';
        dl.style.position = 'absolute';
        dl.style.top = '4px';
        dl.style.right = '4px';
        dl.style.border = 'none';
        dl.style.background = 'rgba(255,255,255,0.9)';
        dl.style.cursor = 'pointer';
        dl.style.padding = '2px 6px';
        dl.addEventListener('click', () => downloadUrl(img.src));

        wrapper.appendChild(thumb);
        wrapper.appendChild(chk);
        wrapper.appendChild(dl);
        grid.appendChild(wrapper);
    });
}

function updateActionsToolbar() {
    const anySelected = getSelectedPageImages().length > 0;
    document.getElementById('actionsToolbar').style.display = anySelected ? 'block' : 'block';
}

function getSelectedPageImages() {
    return Array.from(document.querySelectorAll('#pageImagesGrid input[type="checkbox"]:checked')).map(ch => ch.dataset.src);
}

function toggleAll(checked) {
    document.querySelectorAll('#pageImagesGrid input[type="checkbox"]').forEach(ch => ch.checked = checked);
    updateActionsToolbar();
}

function ensurePickedCTA() {
    const section = document.getElementById('pageImagesSection');
    const empty = document.getElementById('emptyPicked');
    const hasGrid = section && section.style.display !== 'none';
    if (empty) empty.style.display = hasGrid ? 'none' : 'block';
}

function ensureUploadCTA() {
    const previewVisible = document.getElementById('resultSection')?.style.display === 'block';
    const toolbar = document.getElementById('uploadedActionsToolbar');
    const empty = document.getElementById('emptyUploaded');
    if (empty) empty.style.display = previewVisible ? 'none' : 'block';
    if (toolbar) toolbar.style.display = previewVisible ? 'block' : 'none';
}

function paginate(all) {
    const slice = all.slice(pageIndex, pageIndex + PAGE_SIZE);
    pageIndex += PAGE_SIZE;
    return slice;
}

async function runBatch(operation) {
    const targets = getSelectedPageImages();
    if (!targets.length) { updateStatus('Select images first'); return; }
    showLoading(`Processing ${targets.length} image(s)...`);
    // Prepare gallery
    const gallery = document.getElementById('pickedResultsGallery');
    const grid = document.getElementById('pickedResultsGrid');
    const prog = document.getElementById('pickedProgress');
    if (gallery && grid) { gallery.style.display = 'block'; grid.innerHTML = ''; }
    pickedResults = [];
    const single = targets.length === 1;
    __batchProcessing = !single;
    __batchTargetCount = targets.length;
    // If batch, hide single preview section
        if (!single) {
        const sec = document.getElementById('pickedResultSection');
        if (sec) sec.style.display = 'none';
        const prv = document.getElementById('pickedResultPreview');
        if (prv) prv.src = '';
            saveUIState();
    }
    for (const src of targets) {
        const resp = await chrome.runtime.sendMessage({ action: 'processImage', imageSrc: src, operation });
        if (resp && resp.success) {
            lastProcessedDataUrl = resp.result.processedUrl;
            // Show single preview only when one image is processed
            if (single) {
                const prv = document.getElementById('pickedResultPreview');
                const sec = document.getElementById('pickedResultSection');
                if (prv && sec) {
                    prv.src = lastProcessedDataUrl;
                    sec.style.display = 'block';
                    window.switchTab('picked');
                        await saveUIState();
                }
            }
            // Add to gallery
            if (grid) {
                const item = document.createElement('div');
                item.style.border = '1px solid #eee';
                item.style.borderRadius = '4px';
                item.style.background = '#fff';
                item.style.padding = '4px';
                const img = document.createElement('img');
                img.src = lastProcessedDataUrl;
                img.style.width = '100%';
                img.style.height = '80px';
                img.style.objectFit = 'cover';
                const btn = document.createElement('button');
                btn.className = 'btn';
                btn.style.margin = '6px 0 0 0';
                btn.textContent = 'Download';
                btn.addEventListener('click', () => {
                    const a = document.createElement('a');
                    a.href = lastProcessedDataUrl; a.download = `changeimageto-processed-${Date.now()}.png`; a.click();
                });
                item.appendChild(img);
                item.appendChild(btn);
                grid.appendChild(item);
                pickedResults.push(lastProcessedDataUrl);
                if (prog) prog.textContent = `${pickedResults.length}/${targets.length}`;
                    await saveUIState();
            }
        }
    }
    updateStatus('Processing complete');
    hideLoading();
    __batchProcessing = false;
    __batchTargetCount = 0;
    // Ensure preview stays hidden after batch
    const finalizeSec = document.getElementById('pickedResultSection');
    if (finalizeSec) finalizeSec.style.display = 'none';
        await saveUIState();
}

function downloadUrl(url) {
    const a = document.createElement('a');
    a.href = url;
    a.download = '';
    document.body.appendChild(a);
    a.click();
    a.remove();
}

// Toolbar actions
document.getElementById('actRemoveBg').addEventListener('click', () => runBatch('remove-background'));
document.getElementById('actBlurBg').addEventListener('click', () => runBatch('blur-background'));
document.getElementById('actEnhance').addEventListener('click', () => runBatch('enhance-image'));
document.getElementById('actUpscale').addEventListener('click', () => runBatch('upscale-image'));
const actChange = document.getElementById('actChangeBgColor');
if (actChange) actChange.addEventListener('click', async () => {
    const input = document.createElement('input'); input.type = 'color'; input.value = '#ffffff';
    const color = await new Promise((resolve) => { input.addEventListener('input', () => resolve(input.value), { once: true }); input.click(); });
    if (!color) return; await runBatchChangeBgColor(color);
});
document.getElementById('actDownloadOriginal').addEventListener('click', async () => {
    const targets = getSelectedPageImages();
    if (!targets.length) { updateStatus('Select images first'); return; }
    showLoading('Preparing downloads...');
    targets.forEach(downloadUrl);
    hideLoading();
});

// Download all from gallery (data URLs)
const dlAll = document.getElementById('pickedDownloadAll');
if (dlAll) dlAll.addEventListener('click', async () => {
    if (!pickedResults.length) { updateStatus('Nothing to download'); return; }
    // For now, trigger sequential downloads (ZIP would require packaging client-side)
    showLoading('Downloading all...');
    for (const dataUrl of pickedResults) {
        const a = document.createElement('a');
        a.href = dataUrl; a.download = `changeimageto-processed-${Date.now()}.png`; a.click();
        await new Promise(r => setTimeout(r, 150));
    }
    hideLoading();
});

// Close gallery button
const closeGalleryBtn = document.getElementById('pickedCloseGallery');
if (closeGalleryBtn) closeGalleryBtn.addEventListener('click', async () => {
    const gallery = document.getElementById('pickedResultsGallery');
    const grid = document.getElementById('pickedResultsGrid');
    const prog = document.getElementById('pickedProgress');
    if (gallery) gallery.style.display = 'none';
    if (grid) grid.innerHTML = '';
    if (prog) prog.textContent = '';
    pickedResults = [];
    await saveUIState();
});

// Listen for messages from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    switch (request.action) {
        case 'processingStarted':
            showLoading('Processing...');
            break;
        case 'processingEnded':
            hideLoading();
            break;
        case 'showProcessedImage':
            updateStatus(`${request.operation} completed! Preview updated.`);
            if (request.processedUrl) {
                lastProcessedDataUrl = request.processedUrl;
                // Always update Home quick preview (regardless of active tab)
                const homeImg = document.getElementById('homeQuickPreview');
                const homeSec = document.getElementById('homeQuickPreviewSection');
                if (homeImg && homeSec) { homeImg.src = lastProcessedDataUrl; homeSec.style.display = 'block'; saveUIState(); }

                const pickedVisible = document.getElementById('viewPicked')?.style.display === 'block';
                const uploadedVisible = document.getElementById('viewUploaded')?.style.display === 'block';
                if (pickedVisible) {
                    // Suppress single preview when doing or showing bulk results
                    const gallery = document.getElementById('pickedResultsGallery');
                    const grid = document.getElementById('pickedResultsGrid');
                    const bulkActive = (__batchProcessing && __batchTargetCount > 1)
                        || (gallery && gallery.style.display === 'block' && grid && grid.children.length > 0)
                        || (Array.isArray(pickedResults) && pickedResults.length > 1);
                    if (bulkActive) {
                        const sec = document.getElementById('pickedResultSection');
                        if (sec) sec.style.display = 'none';
                        saveUIState();
                        break;
                    }
                    const prv = document.getElementById('pickedResultPreview');
                    const sec = document.getElementById('pickedResultSection');
                    if (prv && sec) { prv.src = lastProcessedDataUrl; sec.style.display = 'block'; saveUIState(); }
                } else if (uploadedVisible) {
                    const imgEl = document.getElementById('resultPreview');
                    if (imgEl) {
                        imgEl.src = lastProcessedDataUrl;
                        const section = document.getElementById('resultSection');
                        if (section) { section.style.display = 'block'; saveUIState(); }
                    }
                }
            }
            break;
        case 'showError':
            updateStatus(`Error: ${request.error}`);
            break;
    }
});

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

function showLoading(text) {
    const overlay = document.getElementById('loadingOverlay');
    const t = document.getElementById('loadingText');
    if (t) t.textContent = text || 'Processing...';
    if (overlay) overlay.style.display = 'flex';
    document.body.classList.add('busy');
    __loadingStartedAt = Date.now();
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    const elapsed = Date.now() - __loadingStartedAt;
    const done = () => { if (overlay) overlay.style.display = 'none'; document.body.classList.remove('busy'); };
    if (elapsed < 400) { setTimeout(done, 400 - elapsed); } else { done(); }
}
