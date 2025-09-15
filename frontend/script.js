const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const processBtn = document.getElementById('process-btn');
const categoryInput = document.getElementById('category');
const catBtns = document.querySelectorAll('.cat-btn');
catBtns.forEach(btn=>{btn.addEventListener('click',()=>{catBtns.forEach(b=>b.classList.remove('active'));btn.classList.add('active');categoryInput.value=btn.dataset.cat;});});
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
    
    // Send to backend analytics endpoint (if available)
    fetch('/api/analytics', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(logEntry)
    }).catch(() => {
        // Silently fail if analytics endpoint is not available
        console.log('Analytics logged:', logEntry);
    });
}

// Log page visit on load
document.addEventListener('DOMContentLoaded', function() {
    const pageType = getPageType();
    logUserAction('page_visit', {
        page_type: pageType,
        url: window.location.href,
        referrer: document.referrer || 'direct'
    });
});

function getPageType() {
    const path = window.location.pathname;
    if (path === '/' || path === '/index.html') return 'remove_background';
    if (path.includes('change-image-background-to-')) return 'color_specific';
    if (path === '/change-image-background.html') return 'color_palette';
    if (path === '/change-color-of-image.html') return 'color_change';
    return 'unknown';
}

function updateCtaText(){
  const isColorPage = !!document.body.getAttribute('data-target-color') || !!document.getElementById('color-palette');
  const isColorChangePage = window.location.pathname === '/change-color-of-image.html';
  
  if (isColorChangePage) {
    processBtn.textContent = 'Change Image Color';
    processBtn.setAttribute('aria-label', 'Change Image Color');
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
  
  if (isColorChangePage) {
    prompt.textContent = 'Press "Change Image Color" to process.';
  } else if (isColorPage) {
    prompt.textContent = 'Press "Change image background" to process.';
  } else {
    prompt.textContent = 'Press "Remove Background" to process.';
  }
}
updateCtaText();

// Color change page specific functionality
if (window.location.pathname === '/change-color-of-image.html') {
    // Initialize sliders
    const hueSlider = document.getElementById('hue-slider');
    const saturationSlider = document.getElementById('saturation-slider');
    const brightnessSlider = document.getElementById('brightness-slider');
    const contrastSlider = document.getElementById('contrast-slider');
    
    const hueValue = document.getElementById('hue-value');
    const saturationValue = document.getElementById('saturation-value');
    const brightnessValue = document.getElementById('brightness-value');
    const contrastValue = document.getElementById('contrast-value');
    
    // Update slider values
    if (hueSlider && hueValue) {
        hueSlider.addEventListener('input', () => {
            hueValue.textContent = hueSlider.value;
        });
    }
    
    if (saturationSlider && saturationValue) {
        saturationSlider.addEventListener('input', () => {
            saturationValue.textContent = saturationSlider.value;
        });
    }
    
    if (brightnessSlider && brightnessValue) {
        brightnessSlider.addEventListener('input', () => {
            brightnessValue.textContent = brightnessSlider.value;
        });
    }
    
    if (contrastSlider && contrastValue) {
        contrastSlider.addEventListener('input', () => {
            contrastValue.textContent = contrastSlider.value;
        });
    }
}

function enableProcess(enabled){
  processBtn.disabled = !enabled;
}

function setOriginalPreview(file){
  const reader = new FileReader();
  reader.onload = () => {
    originalImg.src = reader.result;
    previewSection.hidden = false;
    resultImg.src = '';
    downloadLink.removeAttribute('href');
    const prompt = document.getElementById('process-prompt');
    if(prompt) prompt.hidden = false; updatePromptText();
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

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('drag');
  if(e.dataTransfer.files && e.dataTransfer.files[0]){
    currentFile = e.dataTransfer.files[0];
    setOriginalPreview(currentFile);
    enableProcess(true);
  }
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if(file){
    currentFile = file;
    setOriginalPreview(currentFile);
    enableProcess(true);
  }
});

const form = document.getElementById('upload-form');
// If on a color page, rename CTA
(function(){ const color = document.body.getAttribute('data-target-color'); if(color){ updateCtaText(); processBtn.setAttribute('aria-label','Change image background'); } })();
form.addEventListener('submit', async (e) => {
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
    logDetails.color_type = document.querySelector('.cat-btn.active')?.dataset.cat || 'hue';
    logDetails.hue_shift = document.getElementById('hue-slider')?.value || 0;
    logDetails.saturation = document.getElementById('saturation-slider')?.value || 100;
    logDetails.brightness = document.getElementById('brightness-slider')?.value || 100;
    logDetails.contrast = document.getElementById('contrast-slider')?.value || 100;
  } else {
    logDetails.category = categoryInput.value;
    logDetails.target_color = targetColor;
    logDetails.action_type = targetColor ? 'change_background' : 'remove_background';
  }
  
  logUserAction('processing_started', logDetails);

  try{
    const body = new FormData();
    body.append('file', currentFile);
    
    const apiBase = window.API_BASE || (window.location.hostname === '127.0.0.1' && window.location.port === '8080' ? 'http://127.0.0.1:8000' : 'https://changeimageto.onrender.com');
    let endpoint = '/api/remove-bg';
    
    // Check if this is a color change page
    if (window.location.pathname === '/change-color-of-image.html') {
      endpoint = '/api/change-color';
      // Get the active color type button value
      const activeColorType = document.querySelector('.cat-btn.active')?.dataset.cat || 'hue';
      body.append('color_type', activeColorType);
      body.append('hue_shift', document.getElementById('hue-slider')?.value || 0);
      body.append('saturation', document.getElementById('saturation-slider')?.value || 100);
      body.append('brightness', document.getElementById('brightness-slider')?.value || 100);
      body.append('contrast', document.getElementById('contrast-slider')?.value || 100);
    } else {
      // Background removal or background change
      body.append('category', categoryInput.value);
      var hidden = document.getElementById('bg-color');
      if(hidden) body.append('bg_color', hidden.value);
      var pageColor = document.body.getAttribute('data-target-color');
      if(pageColor) body.append('bg_color', pageColor);
    }

    const res = await fetch(apiBase + endpoint, { method: 'POST', body });
    if(!res.ok){
      const err = await res.text();
      throw new Error(err || 'Failed to process image');
    }
    const blob = await res.blob();
    const objectUrl = URL.createObjectURL(blob);
    resultImg.src = objectUrl;
    downloadLink.href = objectUrl;
    downloadLink.download = `bg-removed-${Date.now()}.png`;
    
    // Log download link creation
    logUserAction('download_link_created', {
      page_type: pageType,
      category: categoryInput.value,
      target_color: targetColor,
      action_type: targetColor ? 'change_background' : 'remove_background',
      filename: downloadLink.download
    });
    
    const prompt = document.getElementById('process-prompt');
    if(prompt) prompt.hidden = true;
    
    // Log successful processing
    let successLogDetails = {
      page_type: pageType,
      output_size: blob.size,
      processing_successful: true
    };
    
    if (pageType === 'color_change') {
      successLogDetails.action_type = 'change_image_color';
      successLogDetails.color_type = document.querySelector('.cat-btn.active')?.dataset.cat || 'hue';
      successLogDetails.hue_shift = document.getElementById('hue-slider')?.value || 0;
      successLogDetails.saturation = document.getElementById('saturation-slider')?.value || 100;
      successLogDetails.brightness = document.getElementById('brightness-slider')?.value || 100;
      successLogDetails.contrast = document.getElementById('contrast-slider')?.value || 100;
    } else {
      successLogDetails.category = categoryInput.value;
      successLogDetails.target_color = targetColor;
      successLogDetails.action_type = targetColor ? 'change_background' : 'remove_background';
    }
    
    logUserAction('processing_completed', successLogDetails);
    
    // make checkerboard solid and edge-to-edge on landing page
    (function(){
      var palette = document.getElementById('color-palette');
      if(!palette) return;
      var checker = document.querySelector('.checkerboard');
      if(checker){
        var hidden = document.getElementById('bg-color');
        var hex = hidden ? hidden.value : '#ffffff';
        checker.style.background = hex;
        checker.style.padding = '0px';
        checker.style.borderRadius = '12px';
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
      errorLogDetails.color_type = document.querySelector('.cat-btn.active')?.dataset.cat || 'hue';
      errorLogDetails.hue_shift = document.getElementById('hue-slider')?.value || 0;
      errorLogDetails.saturation = document.getElementById('saturation-slider')?.value || 100;
      errorLogDetails.brightness = document.getElementById('brightness-slider')?.value || 100;
      errorLogDetails.contrast = document.getElementById('contrast-slider')?.value || 100;
    } else {
      errorLogDetails.category = categoryInput.value;
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

resetBtn.addEventListener('click', () => {
  // Log reset action
  logUserAction('reset_action', {
    page_type: getPageType()
  });
  
  currentFile = null;
  fileInput.value = '';
  originalImg.src = '';
  resultImg.src = '';
  previewSection.hidden = true;
  const prompt = document.getElementById('process-prompt');
  if(prompt) prompt.hidden = true;
  enableProcess(false);
  updateCtaText();
  updatePromptText();
});


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
(function(){
  var pageColor = document.body.getAttribute('data-target-color');
  if(!pageColor) return;
  var checker = document.querySelector('.checkerboard');
  if(checker){
    checker.style.background = pageColor;
    checker.style.padding = '0px';
    checker.style.borderRadius = '12px';
  }
})();


// On color pages, align prompt text with CTA label
(function(){
  var pageColor = document.body.getAttribute('data-target-color');
  if(!pageColor) return;
  var prompt = document.getElementById('process-prompt');
  updatePromptText();
})();


// Landing page color selection behavior
(function(){
  var palette = document.getElementById('color-palette');
  var hidden = document.getElementById('bg-color');
  if(!palette || !hidden) return;
  // Set CTA and prompt
  updateCtaText(); processBtn.setAttribute('aria-label','Change image background');
  var prompt = document.getElementById('process-prompt');
  updatePromptText();
  function setColor(hex){ hidden.value = hex; }
  palette.querySelectorAll('.swatch').forEach(function(btn, i){
    btn.addEventListener('click', function(){
      palette.querySelectorAll('.swatch').forEach(function(b){ b.classList.remove('active'); });
      btn.classList.add('active');
      setColor(btn.getAttribute('data-color'));
      var custom = document.getElementById('custom-color');
      if(custom) custom.value = hidden.value;
    });
    if(i===0) btn.classList.add('active');
  });
  var custom = document.getElementById('custom-color');
  if(custom){ custom.addEventListener('input', function(){ setColor(custom.value); }); }
})();
