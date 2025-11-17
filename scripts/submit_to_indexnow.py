#!/usr/bin/env python3
"""
Submit URLs to IndexNow API for instant search engine indexing.

IndexNow is supported by Bing, Yandex, and other search engines.
This script submits URLs to notify search engines about new or updated content.
"""

import requests
import json
import hashlib
import secrets
import os
from urllib.parse import urlparse

# IndexNow API endpoints
INDEXNOW_ENDPOINTS = [
    "https://www.bing.com/indexnow",
    "https://yandex.com/indexnow",
]

def generate_api_key():
    """Generate a random API key for IndexNow."""
    return secrets.token_urlsafe(32)

def get_key_file_path():
    """Get the path to the IndexNow API key file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, ".indexnow_key")

def get_or_create_api_key():
    """Get existing API key or create a new one."""
    key_file = get_key_file_path()
    
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            return f.read().strip()
    else:
        # Generate new key
        api_key = generate_api_key()
        with open(key_file, 'w') as f:
            f.write(api_key)
        print(f"Generated new IndexNow API key: {api_key}")
        print(f"Key saved to: {key_file}")
        return api_key

def create_key_file(urls):
    """Create the key file that needs to be accessible at the root of the domain."""
    # The key file should be named: {api_key}.txt
    # And placed at: https://www.changeimageto.com/{api_key}.txt
    api_key = get_or_create_api_key()
    key_file_name = f"{api_key}.txt"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(script_dir, '..', 'frontend')
    key_file_path = os.path.join(frontend_dir, key_file_name)
    
    # The key file should contain the API key itself
    with open(key_file_path, 'w') as f:
        f.write(api_key)
    
    print(f"\n⚠️  IMPORTANT: Created key file: {key_file_name}")
    print(f"   This file must be accessible at: https://www.changeimageto.com/{key_file_name}")
    print(f"   Please ensure this file is deployed to your web server root.")
    print(f"   The file contains: {api_key}\n")
    
    return api_key, key_file_name

def submit_to_indexnow(urls, host="www.changeimageto.com"):
    """
    Submit URLs to IndexNow API.
    
    Args:
        urls: List of URLs to submit
        host: The hostname (used for key file location)
    """
    if not urls:
        print("No URLs to submit.")
        return
    
    api_key, key_file_name = create_key_file(urls)
    
    # Prepare the request payload
    payload = {
        "host": host,
        "key": api_key,
        "keyLocation": f"https://{host}/{key_file_name}",
        "urlList": urls
    }
    
    print(f"Submitting {len(urls)} URLs to IndexNow...")
    print(f"API Key: {api_key}")
    print(f"Key Location: https://{host}/{key_file_name}\n")
    
    success_count = 0
    error_count = 0
    
    for endpoint in INDEXNOW_ENDPOINTS:
        try:
            print(f"Submitting to {endpoint}...")
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            # IndexNow returns 200 (OK) or 202 (Accepted) for success
            if response.status_code in [200, 202]:
                print(f"✅ Successfully submitted to {endpoint} (status: {response.status_code})")
                if response.text:
                    print(f"   Response: {response.text}")
                success_count += 1
            else:
                print(f"⚠️  {endpoint} returned status code: {response.status_code}")
                print(f"   Response: {response.text}")
                error_count += 1
        except Exception as e:
            print(f"❌ Error submitting to {endpoint}: {str(e)}")
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Submission Summary:")
    print(f"  Total URLs: {len(urls)}")
    print(f"  Successful endpoints: {success_count}")
    print(f"  Failed endpoints: {error_count}")
    print(f"{'='*60}\n")
    
    if success_count > 0:
        print("✅ URLs have been submitted to IndexNow!")
        print("   Search engines will be notified about these URL changes.")
    else:
        print("❌ Failed to submit to any IndexNow endpoints.")
        print("   Please check your API key file is accessible at the keyLocation URL.")

def main():
    """Main function to submit URLs."""
    # URLs to submit
    urls = [
        "https://www.changeimageto.com/blog/canva-vs-photopea.html",
        "https://www.changeimageto.com/blog/ai-background-removers.html",
        "https://www.changeimageto.com/change-background-to-red.html",
        "https://www.changeimageto.com/remove-background-for-amazon.html",
        "https://www.changeimageto.com/remove-background-for-shopify.html",
        "https://www.changeimageto.com/change-background-to-green.html",
        "https://www.changeimageto.com/change-background-to-purple.html",
        "https://www.changeimageto.com/enhance-image-for-ecommerce.html",
        "https://www.changeimageto.com/remove-text-from-image-for-shopify.html",
        "https://www.changeimageto.com/upscale-image-for-real-estate.html",
        "https://www.changeimageto.com/blur-background-for-portraits.html",
        "https://www.changeimageto.com/upscale-image-for-print.html",
    ]
    
    print("IndexNow URL Submission Script")
    print("=" * 60)
    print(f"Preparing to submit {len(urls)} URLs...\n")
    
    # Validate URLs
    valid_urls = []
    for url in urls:
        try:
            parsed = urlparse(url)
            if parsed.scheme and parsed.netloc:
                valid_urls.append(url)
            else:
                print(f"⚠️  Invalid URL skipped: {url}")
        except Exception as e:
            print(f"⚠️  Error parsing URL {url}: {e}")
    
    if not valid_urls:
        print("❌ No valid URLs to submit.")
        return
    
    print(f"Valid URLs: {len(valid_urls)}")
    for url in valid_urls:
        print(f"  - {url}")
    print()
    
    # Submit to IndexNow
    submit_to_indexnow(valid_urls)

if __name__ == "__main__":
    main()

