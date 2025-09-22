# ChangeImageTo Chrome Extension - Installation Guide

## Quick Installation (Development Mode)

### Step 1: Prepare the Extension
1. Make sure all files are in the `chrome-extension` folder
2. The folder should contain:
   - `manifest.json`
   - `background.js`
   - `content.js`
   - `content.css`
   - `popup.html`
   - `popup.css`
   - `popup.js`
   - `icons/` folder with icon files

### Step 2: Load in Chrome
1. Open Google Chrome
2. Go to `chrome://extensions/`
3. Enable **"Developer mode"** (toggle in top right corner)
4. Click **"Load unpacked"**
5. Select the `chrome-extension` folder
6. The extension should appear in your extensions list

### Step 3: Test the Extension
1. Visit any website with images (e.g., https://unsplash.com)
2. Right-click on any image
3. You should see "Edit with ChangeImageTo" in the context menu
4. Click the extension icon in the toolbar to open the popup

## Testing Checklist

### ✅ Basic Functionality
- [ ] Extension loads without errors in `chrome://extensions/`
- [ ] Extension icon appears in toolbar
- [ ] Right-click context menu shows "Edit with ChangeImageTo"
- [ ] Popup opens when clicking extension icon
- [ ] Images can be selected by clicking on them

### ✅ Image Processing
- [ ] Right-click → Remove Background works
- [ ] Right-click → Blur Background works
- [ ] Right-click → Enhance Image works
- [ ] Right-click → Upscale Image works
- [ ] Processed images can be downloaded

### ✅ Popup Interface
- [ ] Selected images are displayed in popup
- [ ] Quick action buttons work
- [ ] Clear selection button works
- [ ] Loading states are shown during processing
- [ ] Error messages are displayed when needed

## Troubleshooting

### Extension Won't Load
- Check that all files are in the `chrome-extension` folder
- Verify `manifest.json` is valid JSON
- Check Chrome's developer console for errors

### Context Menu Not Appearing
- Make sure you're right-clicking on an actual `<img>` element
- Check that the extension has the required permissions
- Try refreshing the webpage

### API Errors
- Verify the backend URL in `background.js` is correct
- Check that the backend is running and accessible
- Ensure CORS is configured properly on the backend

### Images Not Processing
- Check browser console for error messages
- Verify the image URL is accessible
- Make sure the backend API endpoints are working

## Backend Configuration

The extension uses your existing backend at:
`https://bgremover-backend-121350814881.us-central1.run.app`

### Required CORS Settings
Your backend should allow requests from:
- `chrome-extension://*` (for development)
- `https://www.changeimageto.com` (for production)

### API Endpoints Used
- `POST /api/remove-background`
- `POST /api/blur-background`
- `POST /api/enhance-image`
- `POST /api/upscale-image`

## Production Deployment

### For Chrome Web Store
1. Package the extension as a `.zip` file
2. Submit to Chrome Web Store for review
3. Once approved, users can install from the store

### For Enterprise Distribution
1. Create a `.crx` file using Chrome's developer tools
2. Distribute the `.crx` file to users
3. Users can install by dragging the file to `chrome://extensions/`

## Support

If you encounter issues:
1. Check the browser console for error messages
2. Verify all files are present and correctly formatted
3. Test with different websites and image types
4. Ensure the backend is running and accessible

## Next Steps

After successful testing:
1. **User Feedback**: Test with real users to gather feedback
2. **Feature Enhancement**: Add more editing options
3. **Performance Optimization**: Optimize for large images
4. **Analytics**: Add usage tracking and analytics
5. **Monetization**: Implement premium features
