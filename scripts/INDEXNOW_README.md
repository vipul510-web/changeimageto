# IndexNow URL Submission

This script submits URLs to IndexNow API to notify search engines (Bing, Yandex, etc.) about new or updated content instantly.

## What is IndexNow?

IndexNow is an open protocol that allows websites to instantly inform search engines about URL changes. It's supported by:
- Bing
- Yandex
- Other search engines

## Prerequisites

1. Python 3.6+
2. `requests` library: `pip install requests`

## Setup

1. The script will automatically generate an API key on first run
2. The key file will be created in `frontend/{api_key}.txt`
3. **IMPORTANT**: This key file must be accessible at `https://www.changeimageto.com/{api_key}.txt`

## Usage

```bash
python3 scripts/submit_to_indexnow.py
```

## How It Works

1. **Generate API Key**: On first run, generates a random API key
2. **Create Key File**: Creates `{api_key}.txt` in the frontend directory
3. **Deploy Key File**: You must deploy this file to your web server root
4. **Submit URLs**: Sends POST requests to IndexNow endpoints with your URLs

## Key File Requirements

The IndexNow protocol requires a key file to be accessible at:
```
https://www.changeimageto.com/{api_key}.txt
```

This file must:
- Be accessible via HTTP/HTTPS
- Contain only the API key (no extra content)
- Be in the root directory of your domain

## Current URLs to Submit

The script is configured to submit these 12 URLs:

1. https://www.changeimageto.com/blog/canva-vs-photopea.html
2. https://www.changeimageto.com/blog/ai-background-removers.html
3. https://www.changeimageto.com/change-background-to-red.html
4. https://www.changeimageto.com/remove-background-for-amazon.html
5. https://www.changeimageto.com/remove-background-for-shopify.html
6. https://www.changeimageto.com/change-background-to-green.html
7. https://www.changeimageto.com/change-background-to-purple.html
8. https://www.changeimageto.com/enhance-image-for-ecommerce.html
9. https://www.changeimageto.com/remove-text-from-image-for-shopify.html
10. https://www.changeimageto.com/upscale-image-for-real-estate.html
11. https://www.changeimageto.com/blur-background-for-portraits.html
12. https://www.changeimageto.com/upscale-image-for-print.html

## Steps to Complete Submission

1. **Run the script**:
   ```bash
   python3 scripts/submit_to_indexnow.py
   ```

2. **Deploy the key file**:
   - The script will create a file like `frontend/{api_key}.txt`
   - Deploy this file to your web server root
   - Verify it's accessible at `https://www.changeimageto.com/{api_key}.txt`

3. **Re-run the script** (or it will work on first run if key file is already deployed):
   ```bash
   python3 scripts/submit_to_indexnow.py
   ```

## Verification

After submission, you can verify:
- Check Bing Webmaster Tools for indexing status
- The URLs should be crawled faster than normal
- No errors should appear in the script output

## Notes

- The API key is stored in `scripts/.indexnow_key` for future use
- You can reuse the same API key for multiple submissions
- The key file must remain accessible for IndexNow to work
- You can submit up to 10,000 URLs per request

## Troubleshooting

**Error: Key file not accessible**
- Ensure the key file is deployed to your web server
- Verify the file is accessible via HTTPS
- Check that the file contains only the API key (no extra whitespace)

**Error: 403 Forbidden**
- The key file might not be accessible
- Check file permissions on the server
- Verify the URL matches exactly

**Error: Timeout**
- Check your internet connection
- The IndexNow endpoints might be temporarily unavailable
- Try again later

