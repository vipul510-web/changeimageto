# IndexNow Integration Setup Guide

This guide will help you set up IndexNow for your website to improve SEO by notifying search engines immediately when content is added, updated, or deleted.

## What is IndexNow?

IndexNow is an open protocol that allows websites to notify search engines (Bing, Yandex, and others) immediately when content changes. This helps ensure your latest content is discovered and indexed faster.

For more information, visit: https://www.indexnow.org/

## Setup Steps

### 1. Generate IndexNow API Key

Run the key generation script:

```bash
python scripts/generate_indexnow_key.py
```

This will:
- Generate a random 32-character API key
- Create a key file in the `frontend/` directory (e.g., `{key}.txt`)
- Print instructions for next steps

### 2. Host the Key File

The key file must be accessible at the root of your website. For example:
- `https://your-domain.com/{key}.txt`

**For Vercel deployment:**
- The key file is already in the `frontend/` directory
- Make sure it's included in your deployment
- Verify it's accessible after deployment

**For other hosting:**
- Upload the key file to your web root
- Ensure it's accessible via HTTPS
- The file should contain only the key (no extra whitespace)

### 3. Configure Environment Variables

Add the following environment variables to your backend (Cloud Run, local `.env`, etc.):

```bash
INDEXNOW_KEY=your_generated_key_here
INDEXNOW_SITE_DOMAIN=https://your-domain.com
```

**Important:**
- `INDEXNOW_SITE_DOMAIN` should be your full domain with protocol (e.g., `https://www.example.com`)
- Do NOT include a trailing slash
- The domain should match the domain where your key file is hosted

### 4. Verify Key File Accessibility

After deployment, verify the key file is accessible:

```bash
curl https://your-domain.com/{key}.txt
```

You should see the key as plain text.

### 5. Test the Integration

1. Publish a blog post through the admin interface
2. Check the backend logs for IndexNow submission messages
3. Verify in Bing Webmaster Tools that URLs are being received

## How It Works

When a blog post is published:

1. The post is saved to Google Cloud Storage
2. The system automatically submits the blog post URL to IndexNow
3. IndexNow notifies participating search engines (Bing, Yandex, etc.)
4. Search engines prioritize crawling the submitted URLs

## Manual URL Submission

If you need to manually submit URLs, you can use the IndexNow module:

```python
from backend.indexnow import submit_blog_post, submit_urls

# Submit a single blog post
submit_blog_post(
    site_domain="https://your-domain.com",
    slug="my-blog-post",
    key="your_indexnow_key"
)

# Submit multiple URLs
submit_urls(
    urls=["https://your-domain.com/blog/post1.html", "https://your-domain.com/blog/post2.html"],
    host="your-domain.com",
    key="your_indexnow_key"
)
```

## Troubleshooting

### Key file not accessible
- Verify the file is in the frontend root directory
- Check file permissions
- Ensure your hosting serves `.txt` files correctly
- Verify the URL matches exactly (case-sensitive)

### IndexNow submission fails
- Check that `INDEXNOW_KEY` and `INDEXNOW_SITE_DOMAIN` are set correctly
- Verify the key file is accessible at `https://{domain}/{key}.txt`
- Check backend logs for specific error messages
- Ensure URLs belong to the specified host domain

### URLs not being indexed
- IndexNow notifies search engines but doesn't guarantee indexing
- It may take time for changes to reflect in search results
- Use Bing Webmaster Tools to verify URLs are being received
- Check that your site is properly configured in search engine webmaster tools

## API Response Codes

- `200 OK`: URL submitted successfully
- `202 Accepted`: Request accepted (also success)
- `400 Bad Request`: Invalid format
- `403 Forbidden`: Invalid key (key not found or doesn't match)
- `422 Unprocessable Entity`: URLs don't belong to the host
- `429 Too Many Requests`: Rate limited

## Additional Resources

- [IndexNow Documentation](https://www.indexnow.org/documentation)
- [Bing Webmaster Tools](https://www.bing.com/webmasters)
- [IndexNow Implementation Guide](https://www.bing.com/indexnow/getstarted)

## Notes

- Only submit URLs that have changed since you started using IndexNow
- Don't submit old URLs that haven't changed
- Each crawl counts towards your crawl quota, but IndexNow helps prioritize important URLs
- The integration is non-blocking - if IndexNow fails, blog publishing still succeeds

