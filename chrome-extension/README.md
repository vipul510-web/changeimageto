# ChangeImageTo Chrome Extension

A Chrome extension that allows users to edit images directly from any website using the ChangeImageTo backend services.

## Features

- **Right-click Context Menu**: Edit any image on any website
- **Quick Actions**: Remove background, blur background, enhance image, upscale image
- **Image Selection**: Click on images to select them for batch processing
- **Popup Interface**: Access all editing options through the extension popup
- **Direct Integration**: Uses the existing ChangeImageTo backend APIs

## Installation

### For Development:

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" in the top right
3. Click "Load unpacked" and select the `chrome-extension` folder
4. The extension will be installed and ready to use

### For Production:

1. The extension needs to be packaged and submitted to the Chrome Web Store
2. Or distributed as a `.crx` file for enterprise/manual installation

## Usage

### Method 1: Right-click Context Menu
1. Right-click on any image on any website
2. Select "Edit with ChangeImageTo"
3. Choose your desired operation (Remove Background, Blur Background, etc.)
4. The image will be processed and you can download the result

### Method 2: Image Selection
1. Click on images on a webpage to select them (they'll be highlighted in green)
2. Click the extension icon in the toolbar
3. Use the quick action buttons in the popup
4. Download the processed images

## Technical Details

### Architecture
- **Manifest V3**: Uses the latest Chrome extension manifest format
- **Content Script**: Injects into all web pages to detect and handle images
- **Background Script**: Handles context menus and API communication
- **Popup Interface**: Provides a user-friendly interface for image editing
- **Backend Integration**: Communicates with the existing ChangeImageTo backend

### API Endpoints Used
- `/api/remove-background`
- `/api/blur-background`
- `/api/enhance-image`
- `/api/upscale-image`

### Permissions
- `contextMenus`: To add right-click menu options
- `activeTab`: To interact with the current tab
- `storage`: To store user preferences
- `host_permissions`: To access images on all websites

## Development

### File Structure
```
chrome-extension/
├── manifest.json          # Extension configuration
├── background.js          # Background script (service worker)
├── content.js            # Content script (injected into web pages)
├── content.css           # Styles for content script
├── popup.html            # Popup interface
├── popup.css             # Popup styles
├── popup.js              # Popup functionality
├── icons/                # Extension icons
│   ├── icon16.png
│   ├── icon32.png
│   ├── icon48.png
│   └── icon128.png
└── README.md             # This file
```

### Testing
1. Load the extension in Chrome
2. Visit any website with images
3. Test right-click context menu functionality
4. Test image selection and popup interface
5. Verify backend API integration

### Backend Requirements
The extension requires the backend to have proper CORS configuration to allow requests from Chrome extensions. The backend should allow:
- Origin: `chrome-extension://*` (for development)
- Origin: `https://www.changeimageto.com` (for production)

## Future Enhancements

### Phase 2 Features
- Batch processing of multiple images
- Image format conversion
- Color change operations
- Background replacement
- User preferences and settings
- Usage analytics

### Phase 3 Features
- AI-powered image suggestions
- Cloud storage integration
- Team collaboration features
- Premium features for paid users

## Security Considerations

- All image processing happens through the existing backend
- No images are stored locally in the extension
- CORS policies prevent unauthorized access
- User privacy is maintained (no tracking without consent)

## Support

For issues or questions:
- Visit: https://www.changeimageto.com
- Check the extension popup for quick actions
- Right-click any image for context menu options

## License

This extension is part of the ChangeImageTo platform and follows the same licensing terms.
