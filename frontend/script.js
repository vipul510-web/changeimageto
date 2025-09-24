console.log('script.js: init');
try {
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const processBtn = document.getElementById('process-btn');
const categoryInput = document.getElementById('category');
const catBtns = document.querySelectorAll('.cat-btn');
catBtns.forEach(btn=>{btn.addEventListener('click',()=>{catBtns.forEach(b=>b.classList.remove('active'));btn.classList.add('active'); if(categoryInput){ categoryInput.value=btn.dataset.cat; }});});
const previewSection = document.getElementById('preview-section');
const originalImg = document.getElementById('original-img');
const resultImg = document.getElementById('result-img');
const downloadLink = document.getElementById('download-link');
const resetBtn = document.getElementById('reset-btn');

let currentFile = null;

// Analytics logging functions
function logUserAction(action, details = {}) {
    const logEntry = {
        timestamp: new Date().toISOString(),
        action: action,
        page: window.location.pathname,
        userAgent: navigator.userAgent,
        details: details
    };
    
    // Send to backend analytics endpoint (local dev uses 8000)
    const isLocalFrontend = (['127.0.0.1','localhost'].includes(window.location.hostname)) && window.location.port === '8080';
    const analyticsBase = window.API_BASE || (isLocalFrontend ? 'http://127.0.0.1:8000' : 'https://bgremover-backend-121350814881.us-central1.run.app');
    fetch(analyticsBase + '/api/analytics', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(logEntry)
    }).catch(() => { /* ignore analytics failures */ });
}

// Log page visit on load
document.addEventListener('DOMContentLoaded', function() {
    console.log('script.js: DOMContentLoaded');
    const pageType = getPageType();
    logUserAction('page_visit', {
        page_type: pageType,
        url: window.location.href,
        referrer: document.referrer || 'direct'
    });

    // Inject standard footer link pills across all pages
    try {
      const footer = document.querySelector('footer.container.footer');
      if (footer) {
        const existing = document.querySelector('nav.seo-links');
        if (!existing) {
          const nav = document.createElement('nav');
          nav.className = 'seo-links';
          nav.innerHTML = [
            { href: '/remove-background-from-image.html', label: 'Remove Background from Image' },
            { href: '/enhance-image.html', label: 'Enhance Image' },
            { href: '/upscale-image.html', label: 'AI Image Upscaler' },
            { href: '/remove-text-from-image.html', label: 'Remove Text from Image' },
            { href: '/change-color-of-image.html', label: 'Change color of image online' },
            { href: '/change-image-background.html', label: 'Change image background' },
            { href: '/convert-image-format.html', label: 'Convert image format' },
            { href: '/blur-background.html', label: 'Blur Background' },
            { href: '/bulk-image-resizer.html', label: 'Bulk Image Resizer' },
            { href: '/image-quality-checker.html', label: 'Image Quality Checker' },
          ].map(i => `<a href="${i.href}">${i.label}</a>`).join('');
          // Insert right before footer to match existing structure
          footer.parentElement.insertBefore(nav, footer);
        }
      }
    } catch (e) {
      console.warn('Footer injection failed:', e);
    }
});

function getPageType() {
    const path = window.location.pathname;
    if (path === '/' || path === '/index.html') return 'remove_background';
    if (path.includes('change-image-background-to-')) return 'color_specific';
    if (path === '/change-image-background.html') return 'color_palette';
    if (path === '/change-color-of-image.html') return 'color_change';
    if (path === '/convert-image-format.html') return 'convert_format';
    if (path === '/upscale-image.html') return 'upscale_image';
    if (path === '/blur-background.html') return 'blur_background';
    if (path === '/enhance-image.html') return 'enhance_image';
    if (path === '/remove-text-from-image.html') return 'remove_text';
    
    return 'unknown';
}

function updateCtaText(){
  if(!processBtn) return;
  const isColorPage = !!document.body.getAttribute('data-target-color') || !!document.getElementById('color-palette');
  const isColorChangePage = window.location.pathname === '/change-color-of-image.html';
  const isUpscalePage = window.location.pathname === '/upscale-image.html';
  const isBlurPage = window.location.pathname === '/blur-background.html';
  const isEnhancePage = window.location.pathname === '/enhance-image.html';
  const isRemoveTextPage = window.location.pathname === '/remove-text-from-image.html';
  const isDenoisePage = false;
  
  if (isColorChangePage) {
    processBtn.textContent = 'Change Image Color';
    processBtn.setAttribute('aria-label', 'Change Image Color');
  } else if (isUpscalePage) {
    processBtn.textContent = 'Upscale Image';
    processBtn.setAttribute('aria-label', 'Upscale Image');
  } else if (isBlurPage) {
    processBtn.textContent = 'Blur Background';
    processBtn.setAttribute('aria-label', 'Blur Background');
  } else if (isEnhancePage) {
    processBtn.textContent = 'Enhance Image';
    processBtn.setAttribute('aria-label', 'Enhance Image');
  } else if (isRemoveTextPage) {
    processBtn.textContent = 'Remove Text';
    processBtn.setAttribute('aria-label', 'Remove Text');
  } else if (isColorPage) {
    processBtn.textContent = 'Change image background';
    processBtn.setAttribute('aria-label', 'Change image background');
  } else {
    processBtn.textContent = 'Remove Background';
    processBtn.setAttribute('aria-label', 'Remove Background');
  }
}
function updatePromptText(){
  const prompt = document.getElementById('process-prompt');
  if(!prompt) return;
  const isColorPage = !!document.body.getAttribute('data-target-color') || !!document.getElementById('color-palette');
  const isColorChangePage = window.location.pathname === '/change-color-of-image.html';
  const isUpscalePage = window.location.pathname === '/upscale-image.html';
  const isBlurPage = window.location.pathname === '/blur-background.html';
  const isEnhancePage = window.location.pathname === '/enhance-image.html';
  const isRemoveTextPage = window.location.pathname === '/remove-text-from-image.html';
  const isDenoisePage = false;
  
  if (isColorChangePage) {
    prompt.textContent = 'Press "Change Image Color" to process.';
  } else if (isUpscalePage) {
    prompt.textContent = 'Press "Upscale Image" to process.';
  } else if (isBlurPage) {
    prompt.textContent = 'Press "Blur Background" to process.';
  } else if (isEnhancePage) {
    prompt.textContent = 'Press "Enhance Image" to process.';
  } else if (isRemoveTextPage) {
    prompt.textContent = 'Press "Remove Text" to process.';
  } else if (isColorPage) {
    prompt.textContent = 'Press "Change image background" to process.';
  } else {
    prompt.textContent = 'Press "Remove Background" to process.';
  }
}
updateCtaText();

// Color change page specific functionality
if (window.location.pathname === '/change-color-of-image.html') {
    // Wait for DOM to be ready
    document.addEventListener('DOMContentLoaded', function() {
        // Initialize sliders
        const hueSlider = document.getElementById('hue-slider');
        const saturationSlider = document.getElementById('saturation-slider');
        const brightnessSlider = document.getElementById('brightness-slider');
        const contrastSlider = document.getElementById('contrast-slider');
        
        const hueValue = document.getElementById('hue-value');
        const saturationValue = document.getElementById('saturation-value');
        const brightnessValue = document.getElementById('brightness-value');
        const contrastValue = document.getElementById('contrast-value');
        
        console.log('Sliders found:', { hueSlider, saturationSlider, brightnessSlider, contrastSlider });
        console.log('Value elements found:', { hueValue, saturationValue, brightnessValue, contrastValue });
        
        // Update slider values
        if (hueSlider && hueValue) {
            hueSlider.addEventListener('input', () => {
                hueValue.textContent = hueSlider.value + '°';
                console.log('Hue changed to:', hueSlider.value);
            });
        }
        
        if (saturationSlider && saturationValue) {
            saturationSlider.addEventListener('input', () => {
                saturationValue.textContent = saturationSlider.value + '%';
                console.log('Saturation changed to:', saturationSlider.value);
            });
        }
        
        if (brightnessSlider && brightnessValue) {
            brightnessSlider.addEventListener('input', () => {
                brightnessValue.textContent = brightnessSlider.value + '%';
                console.log('Brightness changed to:', brightnessSlider.value);
            });
        }
        
        if (contrastSlider && contrastValue) {
            contrastSlider.addEventListener('input', () => {
                contrastValue.textContent = contrastSlider.value + '%';
                console.log('Contrast changed to:', contrastSlider.value);
            });
        }
    
        // Preset buttons functionality
        const presetButtons = document.querySelectorAll('.preset-btn');
        presetButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const preset = btn.dataset.preset;
                applyPreset(preset);
                
                // Update active state
                presetButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });
        
        function applyPreset(preset) {
            switch(preset) {
                case 'vibrant':
                    hueSlider.value = 0;
                    saturationSlider.value = 150;
                    brightnessSlider.value = 110;
                    contrastSlider.value = 120;
                    break;
                case 'muted':
                    hueSlider.value = 0;
                    saturationSlider.value = 50;
                    brightnessSlider.value = 90;
                    contrastSlider.value = 80;
                    break;
                case 'warm':
                    hueSlider.value = 30;
                    saturationSlider.value = 120;
                    brightnessSlider.value = 110;
                    contrastSlider.value = 110;
                    break;
                case 'cool':
                    hueSlider.value = -30;
                    saturationSlider.value = 120;
                    brightnessSlider.value = 110;
                    contrastSlider.value = 110;
                    break;
                case 'grayscale':
                    hueSlider.value = 0;
                    saturationSlider.value = 0;
                    brightnessSlider.value = 100;
                    contrastSlider.value = 120;
                    break;
                case 'reset':
                    hueSlider.value = 0;
                    saturationSlider.value = 100;
                    brightnessSlider.value = 100;
                    contrastSlider.value = 100;
                    break;
            }
            
            // Update display values
            hueValue.textContent = hueSlider.value + '°';
            saturationValue.textContent = saturationSlider.value + '%';
            brightnessValue.textContent = brightnessSlider.value + '%';
            contrastValue.textContent = contrastSlider.value + '%';
        }
    });
}

function enableProcess(enabled){
  if(processBtn) processBtn.disabled = !enabled;
}

function setOriginalPreview(file){
  const reader = new FileReader();
  reader.onload = () => {
    originalImg.src = reader.result;
    originalImg.style.display = 'block';
    if (previewSection) {
      previewSection.hidden = false;
    }
    
    // Hide upload prompt and show image
    const uploadPrompt = document.getElementById('upload-prompt');
    if (uploadPrompt) uploadPrompt.style.display = 'none';
    
    // Show the image columns
    const imageColumns = document.querySelectorAll('.image-column');
    imageColumns.forEach(col => col.style.display = 'block');
    
    resultImg.src = '';
    downloadLink.removeAttribute('href');
    const prompt = document.getElementById('process-prompt');
    if(prompt) prompt.style.display = 'block';
    const resultWrap = document.getElementById('result-wrapper');
    if(resultWrap) resultWrap.hidden = true;
    
    // Log file upload
    logUserAction('file_uploaded', {
      filename: file.name,
      file_size: file.size,
      file_type: file.type,
      page_type: getPageType()
    });
  };
  reader.readAsDataURL(file);
}

if (dropzone) {
  dropzone.addEventListener('click', () => { 
    const fi = document.getElementById('file-input');
    console.log('dropzone click', !!fi);
    if (fi) {
      if (typeof fi.showPicker === 'function') {
        try { fi.showPicker(); return; } catch(_) {}
      }
      fi.click();
    }
  });
  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag');
    if(e.dataTransfer.files && e.dataTransfer.files[0]){
      console.log('drop event got file');
      currentFile = e.dataTransfer.files[0];
      setOriginalPreview(currentFile);
      enableProcess(true);
    }
  });
}

// Fallback: delegate click anywhere inside the uploader to open the file chooser
document.addEventListener('click', (e) => {
  const dz = e.target.closest && e.target.closest('#dropzone');
  if (dz && fileInput) {
    e.preventDefault();
    fileInput.click();
  }
});

if (fileInput) {
  fileInput.addEventListener('change', (e) => {
    console.log('fileInput change fired');
    const file = e.target.files[0];
    if(file){
      currentFile = file;
      setOriginalPreview(currentFile);
      enableProcess(true);
    }
  });
}

// Defensive: also listen for change events bubbled from any #file-input that might be re-rendered
document.addEventListener('change', (e) => {
  const target = e.target;
  if (target && target.id === 'file-input' && target.files && target.files[0]) {
    console.log('file-input delegated change');
    currentFile = target.files[0];
    setOriginalPreview(currentFile);
    enableProcess(true);
  }
});

const form = document.getElementById('upload-form');
// If on a color page, rename CTA
(function(){ const color = document.body.getAttribute('data-target-color'); if(color){ updateCtaText(); processBtn.setAttribute('aria-label','Change image background'); } })();
if (form) form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if(!currentFile) return;
  enableProcess(false);
  processBtn.textContent = 'Processing…';

  // Log processing start
  const pageType = getPageType();
  const targetColor = document.body.getAttribute('data-target-color') || document.getElementById('bg-color')?.value;
  
  let logDetails = {
    page_type: pageType,
    filename: currentFile.name,
    file_size: currentFile.size
  };
  
  if (pageType === 'color_change') {
    logDetails.action_type = 'change_image_color';
    logDetails.color_type = 'hue';
    logDetails.hue_shift = document.getElementById('hue-slider')?.value || 0;
    logDetails.saturation = document.getElementById('saturation-slider')?.value || 100;
    logDetails.brightness = document.getElementById('brightness-slider')?.value || 100;
    logDetails.contrast = document.getElementById('contrast-slider')?.value || 100;
  } else if (pageType === 'upscale_image') {
    logDetails.action_type = 'upscale_image';
    logDetails.scale_factor = document.getElementById('scale-factor')?.value || '2';
    logDetails.method = document.getElementById('upscale-method')?.value || 'lanczos';
  } else if (pageType === 'blur_background') {
    logDetails.action_type = 'blur_background';
    logDetails.blur_radius = document.getElementById('blur-radius')?.value || '12';
  } else if (pageType === 'enhance_image') {
    logDetails.action_type = 'enhance_image';
    logDetails.preset = document.getElementById('enhance-preset')?.value || 'balanced';
  } else if (pageType === 'remove_text') {
    logDetails.action_type = 'remove_text';
  
  } else {
    logDetails.category = categoryInput?.value || 'unknown';
    logDetails.target_color = targetColor;
    logDetails.action_type = targetColor ? 'change_background' : 'remove_background';
  }
  
  logUserAction('processing_started', logDetails);

  try{
    const body = new FormData();
    body.append('file', currentFile);
    
    const isLocalFrontend = (['127.0.0.1','localhost'].includes(window.location.hostname)) && window.location.port === '8080';
    const apiBase = window.API_BASE || (isLocalFrontend ? 'http://127.0.0.1:8000' : 'https://bgremover-backend-121350814881.us-central1.run.app');
    let endpoint = '/api/remove-bg';
    
    console.log('API Base:', apiBase);
    console.log('Current location:', window.location.hostname, window.location.port);
    
    // Check if this is a color change page
    if (window.location.pathname === '/change-color-of-image.html') {
      endpoint = '/api/change-color';
      // Always use 'hue' as color_type since we're doing comprehensive color adjustment
      const hueSlider = document.getElementById('hue-slider');
      const saturationSlider = document.getElementById('saturation-slider');
      const brightnessSlider = document.getElementById('brightness-slider');
      const contrastSlider = document.getElementById('contrast-slider');
      
      console.log('Slider elements found:', {
        hueSlider: !!hueSlider,
        saturationSlider: !!saturationSlider,
        brightnessSlider: !!brightnessSlider,
        contrastSlider: !!contrastSlider
      });
      
      const hueShift = hueSlider?.value || 0;
      const saturation = saturationSlider?.value || 100;
      const brightness = brightnessSlider?.value || 100;
      const contrast = contrastSlider?.value || 100;
      
      console.log('Color change values:', { hueShift, saturation, brightness, contrast });
      
      body.append('color_type', 'hue');
      body.append('hue_shift', hueShift);
      body.append('saturation', saturation);
      body.append('brightness', brightness);
      body.append('contrast', contrast);
      
      console.log('FormData contents:');
      for (let [key, value] of body.entries()) {
        console.log(key, ':', value);
      }
    } else if (window.location.pathname === '/upscale-image.html') {
      // Upscaling page
      endpoint = '/api/upscale-image';
      const scaleFactor = document.getElementById('scale-factor')?.value || '2';
      const method = document.getElementById('upscale-method')?.value || 'lanczos';
      
      console.log('Upscaling values:', { scaleFactor, method });
      
      body.append('scale_factor', scaleFactor);
      body.append('method', method);
      
      console.log('FormData contents:');
      for (let [key, value] of body.entries()) {
        console.log(key, ':', value);
      }
    } else if (window.location.pathname === '/blur-background.html') {
      endpoint = '/api/blur-background';
      const radius = document.getElementById('blur-radius')?.value || '12';
      body.append('blur_radius', radius);
    } else if (window.location.pathname === '/enhance-image.html') {
      endpoint = '/api/enhance-image';
      const preset = document.getElementById('enhance-preset')?.value || 'balanced';
      // Map preset to parameters server expects
      if (preset === 'soft') { body.append('sharpen', '1.0'); body.append('contrast', '105'); body.append('brightness', '100'); }
      if (preset === 'balanced') { body.append('sharpen', '1.3'); body.append('contrast', '110'); body.append('brightness', '102'); }
      if (preset === 'strong') { body.append('sharpen', '1.6'); body.append('contrast', '120'); body.append('brightness', '105'); }
    } else if (window.location.pathname === '/remove-text-from-image.html') {
      endpoint = '/api/remove-text';
    
    } else {
      // Background removal or background change
      body.append('category', categoryInput?.value || 'product');
      var hidden = document.getElementById('bg-color');
      if(hidden) body.append('bg_color', hidden.value);
      // If a swatch is active, prefer that color
      var activeSwatch = document.querySelector('#color-palette .swatch.active');
      if(activeSwatch){ body.append('bg_color', activeSwatch.getAttribute('data-color')); }
      var pageColor = document.body.getAttribute('data-target-color');
      if(pageColor) body.append('bg_color', pageColor);
    }

        console.log('Making API call to:', apiBase + endpoint);
        const res = await fetch(apiBase + endpoint, { method: 'POST', body });
        console.log('API response status:', res.status);
        console.log('API response headers:', res.headers);
        if(!res.ok){
          const err = await res.text();
          console.error('API error:', err);
          throw new Error(err || 'Failed to process image');
        }
        const blob = await res.blob();
        console.log('Response blob size:', blob.size, 'bytes');
        const objectUrl = URL.createObjectURL(blob);
        resultImg.src = objectUrl;
        downloadLink.href = objectUrl;
        // Set a sensible filename based on page type
        if (pageType === 'upscale_image') {
          downloadLink.download = `upscaled-${Date.now()}.png`;
        } else if (pageType === 'color_change') {
          downloadLink.download = `color-changed-${Date.now()}.png`;
        } else if (pageType === 'blur_background') {
          downloadLink.download = `blur-background-${Date.now()}.png`;
        } else if (pageType === 'enhance_image') {
          downloadLink.download = `enhanced-${Date.now()}.png`;
        } else if (pageType === 'remove_text') {
          downloadLink.download = `text-removed-${Date.now()}.png`;
        
        } else {
          downloadLink.download = `bg-removed-${Date.now()}.png`;
        }
        try {
          // Convert blob to DataURL so it persists across navigation (Blob URLs are per-document)
          const fr = new FileReader();
          fr.onload = () => {
            try {
              sessionStorage.setItem('preloadImageDataURL', fr.result);
              sessionStorage.setItem('preloadImageName', downloadLink.download || `image-${Date.now()}.png`);
              sessionStorage.removeItem('preloadImageURL');
            } catch(_) {}
          };
          fr.readAsDataURL(blob);
          var toConv = document.getElementById('to-converter');
          if (toConv) toConv.style.display = 'inline-block';
        } catch(_){}
    
    // Log download link creation
    logUserAction('download_link_created', {
      page_type: pageType,
      category: categoryInput?.value || 'unknown',
      target_color: targetColor,
      action_type: targetColor ? 'change_background' : 'remove_background',
      filename: downloadLink.download
    });
    
    const prompt = document.getElementById('process-prompt');
    if(prompt) {
      prompt.hidden = true;
      prompt.style.display = 'none';
    }
    
    // Log successful processing
    let successLogDetails = {
      page_type: pageType,
      output_size: blob.size,
      processing_successful: true
    };
    
    if (pageType === 'color_change') {
      successLogDetails.action_type = 'change_image_color';
      successLogDetails.color_type = 'hue';
      successLogDetails.hue_shift = document.getElementById('hue-slider')?.value || 0;
      successLogDetails.saturation = document.getElementById('saturation-slider')?.value || 100;
      successLogDetails.brightness = document.getElementById('brightness-slider')?.value || 100;
      successLogDetails.contrast = document.getElementById('contrast-slider')?.value || 100;
    } else if (pageType === 'upscale_image') {
      successLogDetails.action_type = 'upscale_image';
      successLogDetails.scale_factor = document.getElementById('scale-factor')?.value || '2';
      successLogDetails.method = document.getElementById('upscale-method')?.value || 'lanczos';
    } else if (pageType === 'blur_background') {
      successLogDetails.action_type = 'blur_background';
      successLogDetails.blur_radius = document.getElementById('blur-radius')?.value || '12';
    } else if (pageType === 'enhance_image') {
      successLogDetails.action_type = 'enhance_image';
      successLogDetails.preset = document.getElementById('enhance-preset')?.value || 'balanced';
    } else if (pageType === 'remove_text') {
      successLogDetails.action_type = 'remove_text';
    
    } else {
      successLogDetails.category = categoryInput?.value || 'unknown';
      successLogDetails.target_color = targetColor;
      successLogDetails.action_type = targetColor ? 'change_background' : 'remove_background';
    }
    
    logUserAction('processing_completed', successLogDetails);
    
    // make checkerboard solid and edge-to-edge on landing page
    (function(){
      var palette = document.getElementById('color-palette');
      var pageColor = document.body.getAttribute('data-target-color');
      var checker = document.querySelector('.checkerboard');
      
      if(checker){
        var hex = '#ffffff'; // default
        
        if(palette) {
          // General background page with color palette
          var hidden = document.getElementById('bg-color');
          hex = hidden ? hidden.value : '#ffffff';
        } else if(pageColor) {
          // Color-specific page
          hex = pageColor;
        }
        
        checker.style.background = hex;
        checker.style.padding = '0px';
        checker.style.borderRadius = '12px';
        
        // Add processed class for CSS specificity
        checker.classList.add('processed');
        
        // Force the styles to be applied
        checker.style.setProperty('padding', '0px', 'important');
        checker.style.setProperty('background', hex, 'important');
      }
    })();
    const resultWrap = document.getElementById('result-wrapper');
    if(resultWrap) resultWrap.hidden = false;
  }catch(err){
    // Log processing error
    let errorLogDetails = {
      page_type: pageType,
      error_message: err.message || err,
      processing_successful: false
    };
    
    if (pageType === 'color_change') {
      errorLogDetails.action_type = 'change_image_color';
      errorLogDetails.color_type = 'hue';
      errorLogDetails.hue_shift = document.getElementById('hue-slider')?.value || 0;
      errorLogDetails.saturation = document.getElementById('saturation-slider')?.value || 100;
      errorLogDetails.brightness = document.getElementById('brightness-slider')?.value || 100;
      errorLogDetails.contrast = document.getElementById('contrast-slider')?.value || 100;
    } else if (pageType === 'upscale_image') {
      errorLogDetails.action_type = 'upscale_image';
      errorLogDetails.scale_factor = document.getElementById('scale-factor')?.value || '2';
      errorLogDetails.method = document.getElementById('upscale-method')?.value || 'lanczos';
    } else if (pageType === 'blur_background') {
      errorLogDetails.action_type = 'blur_background';
      errorLogDetails.blur_radius = document.getElementById('blur-radius')?.value || '12';
    } else if (pageType === 'enhance_image') {
      errorLogDetails.action_type = 'enhance_image';
      errorLogDetails.preset = document.getElementById('enhance-preset')?.value || 'balanced';
    } else if (pageType === 'remove_text') {
      errorLogDetails.action_type = 'remove_text';
    
    } else {
      errorLogDetails.category = categoryInput?.value || 'unknown';
      errorLogDetails.target_color = targetColor;
      errorLogDetails.action_type = targetColor ? 'change_background' : 'remove_background';
    }
    
    logUserAction('processing_error', errorLogDetails);
    
    alert('Error: ' + (err.message || err));
  }finally{
    updateCtaText();
    enableProcess(true);
  }
});

// ------------------------------
// Convert format page integration
// ------------------------------
if (window.location.pathname === '/convert-image-format.html') {
  document.addEventListener('DOMContentLoaded', function(){
    const dz = document.getElementById('convert-dropzone');
    const input = document.getElementById('convert-file');
    const btn = document.getElementById('convert-btn');
    const orig = document.getElementById('convert-original');
    const resImg = document.getElementById('convert-result');
    const resWrap = document.getElementById('convert-result-wrapper');
    const prompt = document.getElementById('convert-prompt');
    const preview = document.getElementById('convert-preview');
    const downloadA = document.getElementById('convert-download');
    const resetC = document.getElementById('convert-reset');
    const target = document.getElementById('target-format');
    const keepT = document.getElementById('keep-transparent');
    const quality = document.getElementById('quality');
    const openA = document.getElementById('convert-open');

    let file = null;
    function setPreview(f){
      const r = new FileReader();
      r.onload = ()=>{ orig.src = r.result; preview.hidden = false; btn.disabled = false; };
      r.readAsDataURL(f);
    }
    dz.addEventListener('click', ()=> input.click());
    dz.addEventListener('dragover', e=>{e.preventDefault(); dz.classList.add('drag');});
    dz.addEventListener('dragleave', ()=> dz.classList.remove('drag'));
    dz.addEventListener('drop', e=>{e.preventDefault(); dz.classList.remove('drag'); if(e.dataTransfer.files[0]){ file = e.dataTransfer.files[0]; setPreview(file);} });
    input.addEventListener('change', e=>{ if(e.target.files[0]){ file = e.target.files[0]; setPreview(file);} });

    document.getElementById('convert-form').addEventListener('submit', async function(e){
      e.preventDefault(); if(!file) return; btn.disabled = true; btn.textContent = 'Converting…';
      const body = new FormData();
      body.append('file', file);
      body.append('target_format', target.value);
      body.append('transparent', keepT.checked ? 'true' : 'false');
      body.append('quality', quality.value);
      const apiBase = window.API_BASE || (window.location.hostname === '127.0.0.1' && window.location.port === '8080' ? 'http://127.0.0.1:8000' : 'https://bgremover-backend-121350814881.us-central1.run.app');
      try {
        const res = await fetch(apiBase + '/api/convert-format', { method: 'POST', body });
        if(!res.ok){ throw new Error(await res.text()); }
        const blob = await res.blob();
        // For formats browsers can't preview (ICO, PPM, PGM, TIFF), skip inline preview and just enable download
        const ct = res.headers.get('content-type') || '';
        const nonPreview = ['image/x-icon','image/tiff','image/x-portable-pixmap','image/x-portable-graymap'];
        const url = URL.createObjectURL(blob);
        const ext = target.value === 'jpg' ? 'jpg' : target.value;
        downloadA.href = url;
        downloadA.download = `converted-${Date.now()}.${ext}`;
        resWrap.hidden = false;
        prompt.style.display = 'none';
        if (!nonPreview.includes(ct)) {
          resImg.style.display = 'block';
          resImg.src = url;
        } else {
          // Hide unsupported preview types to avoid broken tiny icon
          resImg.removeAttribute('src');
          resImg.style.display = 'none';
          // Optionally show a text note
          if (!document.getElementById('convert-note')) {
            const note = document.createElement('p');
            note.id = 'convert-note';
            note.className = 'prompt';
            note.textContent = 'Preview not supported in browser for this format. Use Download to view the file.';
            resWrap.parentElement.insertBefore(note, resWrap.nextSibling);
          }
        }
        if (openA){ openA.href = url; openA.target = '_blank'; openA.style.display = 'inline-block'; }
        logUserAction('convert_completed', { target_format: target.value, transparent: keepT.checked, size: blob.size });
      } catch (err) {
        alert('Error: ' + (err.message || err));
        logUserAction('convert_error', { message: err.message || String(err) });
      } finally {
        btn.disabled = false; btn.textContent = 'Convert Image';
      }
    });

    resetC.addEventListener('click', function(){
      file = null; input.value = ''; orig.src = ''; resImg.src = ''; preview.hidden = true; resWrap.hidden = true; prompt.style.display = 'block'; btn.disabled = true;
      if (openA){ openA.removeAttribute('href'); openA.style.display = 'none'; }
    });

    // Preset buttons for GIMP/Inkscape
    const convertPresets = document.querySelectorAll('[data-convert-preset]');
    convertPresets.forEach(btn => {
      btn.addEventListener('click', function(){
        const preset = btn.getAttribute('data-convert-preset');
        if (preset === 'gimp') {
          target.value = 'png';
          keepT.checked = true;
          quality.value = '90';
        } else if (preset === 'inkscape-png') {
          target.value = 'png';
          keepT.checked = true;
          quality.value = '90';
        } else if (preset === 'inkscape-webp') {
          target.value = 'webp';
          keepT.checked = true;
          quality.value = '90';
        }
        logUserAction('convert_preset_selected', { preset, target_format: target.value, transparent: keepT.checked, quality: quality.value });
      });
    });

    // Auto preload from sessionStorage
    try {
      const preloadDataURL = sessionStorage.getItem('preloadImageDataURL');
      const preloadName = sessionStorage.getItem('preloadImageName') || 'image.png';
      if (preloadDataURL) {
        fetch(preloadDataURL).then(r=>r.blob()).then(b=>{
          const f = new File([b], preloadName, { type: b.type || 'image/png' });
          file = f; setPreview(file);
        }).catch(()=>{});
      }
    } catch(_){ }
  });
}

if (resetBtn) resetBtn.addEventListener('click', () => {
  // Log reset action
  logUserAction('reset_action', {
    page_type: getPageType()
  });
  
  currentFile = null;
  fileInput.value = '';
  originalImg.src = '';
  originalImg.style.display = 'none';
  resultImg.src = '';
  
  // Show upload prompt
  const uploadPrompt = document.getElementById('upload-prompt');
  if (uploadPrompt) uploadPrompt.style.display = 'block';
  
  // Hide image columns initially
  const imageColumns = document.querySelectorAll('.image-column');
  imageColumns.forEach(col => col.style.display = 'none');
  
  const prompt = document.getElementById('process-prompt');
  if(prompt) prompt.style.display = 'none';
  const resultWrap = document.getElementById('result-wrapper');
  if(resultWrap) resultWrap.hidden = true;
  
  enableProcess(false);
  updateCtaText();
  updatePromptText();
});

} catch (e) {
  console.error('script.js init error:', e);
}


// Color-page composition: if body has data-target-color, render solid background
(function(){
  var color = document.body.getAttribute("data-target-color");
  if(!color) return;
  async function toColorBackground(url){
    return new Promise(function(resolve, reject){
      var img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = function(){
        try{
          var canvas = document.createElement("canvas");
          canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
          var ctx = canvas.getContext("2d");
          ctx.fillStyle = color; ctx.fillRect(0,0,canvas.width,canvas.height);
          ctx.drawImage(img,0,0);
          canvas.toBlob(function(b){ if(b) resolve(b); else reject(new Error("blob failed")); }, "image/png");
        }catch(e){ reject(e); }
      };
      img.onerror = reject;
      img.src = url;
    });
  }
  form.addEventListener('submit', function(){
    setTimeout(async function(){
      if(!downloadLink.href) return;
      var backendColored = !!document.body.getAttribute('data-target-color');
      if(backendColored){ return; }
      try{
        var blob = await toColorBackground(downloadLink.href);
        var coloredUrl = URL.createObjectURL(blob);
        resultImg.src = coloredUrl;
        downloadLink.href = coloredUrl;
        var hex = color.replace('#','');
        downloadLink.download = 'bg-' + hex + '-' + Date.now() + '.png';
      }catch(err){ console.warn('Color composite failed', err); }
    }, 0);
  });
})();


// If on a color page, make the preview background solid and edge-to-edge
// This will be applied after processing is complete


// On color pages, align prompt text with CTA label
(function(){
  var pageColor = document.body.getAttribute('data-target-color');
  if(!pageColor) return;
  var prompt = document.getElementById('process-prompt');
  updatePromptText();
})();


// Landing page color selection behavior
document.addEventListener('DOMContentLoaded', function(){
  var palette = document.getElementById('color-palette');
  var hidden = document.getElementById('bg-color');
  if(!palette || !hidden){ console.log('palette or hidden not found'); return; }
  // Set CTA and prompt
  updateCtaText(); if(processBtn) processBtn.setAttribute('aria-label','Change image background');
  updatePromptText();
  function setColor(hex){ hidden.value = hex; }
  // Fresh, simple event delegation for palette clicks
  palette.addEventListener('click', function(e){
    var btn = e.target.closest && e.target.closest('.swatch');
    if(!btn || !palette.contains(btn)) return;
    e.preventDefault();
    palette.querySelectorAll('.swatch').forEach(function(b){ b.classList.remove('active'); });
    btn.classList.add('active');
    setColor(btn.getAttribute('data-color'));
    var custom = document.getElementById('custom-color');
    if(custom) custom.value = hidden.value;
    console.log('bg_color set', hidden.value);
  });
  // Initialize first swatch as active if none selected
  var first = palette.querySelector('.swatch');
  if(first && !palette.querySelector('.swatch.active')){
    first.classList.add('active');
    setColor(first.getAttribute('data-color'));
  }
  var custom = document.getElementById('custom-color');
  if(custom){ custom.addEventListener('input', function(){ setColor(custom.value); }); }
});

// Extra defensive: global delegate to ensure swatch selection always works, even if previous binding missed
document.addEventListener('click', function(e){
  var palette = document.getElementById('color-palette');
  var hidden = document.getElementById('bg-color');
  if(!palette || !hidden) return;
  var sw = e.target.closest && e.target.closest('.swatch');
  if(sw && palette.contains(sw)){
    e.preventDefault();
    palette.querySelectorAll('.swatch').forEach(function(b){ b.classList.remove('active'); b.style.outline='none'; });
    sw.classList.add('active');
    sw.style.outline = '2px solid var(--accent)';
    hidden.value = sw.getAttribute('data-color');
    var custom = document.getElementById('custom-color');
    if(custom) custom.value = hidden.value;
    console.log('global delegate set bg_color', hidden.value);
  }
});