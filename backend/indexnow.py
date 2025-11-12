"""
IndexNow integration for notifying search engines about content updates.

IndexNow is a protocol that allows websites to notify search engines immediately
when content is added, updated, or deleted. This helps improve SEO by ensuring
search engines are aware of changes quickly.

For more information: https://www.indexnow.org/
"""

import os
import logging
import requests
from typing import List, Optional
import secrets
import string

logger = logging.getLogger(__name__)

# IndexNow API endpoint
INDEXNOW_API_URL = "https://api.indexnow.org/IndexNow"

# Generate a random IndexNow API key
def generate_indexnow_key(length: int = 32) -> str:
    """
    Generate a random IndexNow API key.
    
    Args:
        length: Length of the key (default: 32 characters)
        
    Returns:
        A random alphanumeric key
    """
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def submit_urls(
    urls: List[str],
    host: str,
    key: str,
    key_location: Optional[str] = None
) -> bool:
    """
    Submit URLs to IndexNow API.
    
    Args:
        urls: List of full URLs to submit (e.g., ["https://example.com/blog/post.html"])
        host: The host domain (e.g., "www.example.com" or "example.com")
        key: The IndexNow API key
        key_location: Optional location of the key file (e.g., "https://example.com/{key}.txt")
                     If not provided, defaults to root: "https://{host}/{key}.txt"
    
    Returns:
        True if submission was successful, False otherwise
    """
    if not urls:
        logger.warning("No URLs provided for IndexNow submission")
        return False
    
    # Build key location URL if not provided
    if not key_location:
        # Ensure host doesn't have protocol
        host_clean = host.replace("https://", "").replace("http://", "").strip("/")
        key_location = f"https://{host_clean}/{key}.txt"
    
    # Prepare the request payload
    payload = {
        "host": host.replace("https://", "").replace("http://", "").strip("/"),
        "key": key,
        "keyLocation": key_location,
        "urlList": urls
    }
    
    try:
        response = requests.post(
            INDEXNOW_API_URL,
            json=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully submitted {len(urls)} URLs to IndexNow")
            return True
        elif response.status_code == 202:
            logger.info(f"IndexNow accepted {len(urls)} URLs (202 Accepted)")
            return True
        elif response.status_code == 400:
            logger.error(f"IndexNow submission failed: Bad request - {response.text}")
            return False
        elif response.status_code == 403:
            logger.error(f"IndexNow submission failed: Invalid key - {response.text}")
            return False
        elif response.status_code == 422:
            logger.error(f"IndexNow submission failed: URLs don't belong to host - {response.text}")
            return False
        elif response.status_code == 429:
            logger.warning(f"IndexNow submission rate limited - {response.text}")
            return False
        else:
            logger.error(f"IndexNow submission failed with status {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error submitting URLs to IndexNow: {str(e)}")
        return False


def submit_blog_post(site_domain: str, slug: str, key: str, key_location: Optional[str] = None) -> bool:
    """
    Submit a single blog post URL to IndexNow.
    
    Args:
        site_domain: The site domain (e.g., "https://www.example.com")
        slug: The blog post slug (e.g., "my-blog-post")
        key: The IndexNow API key
        key_location: Optional location of the key file
    
    Returns:
        True if submission was successful, False otherwise
    """
    # Ensure site_domain doesn't have trailing slash
    site_domain = site_domain.rstrip("/")
    
    # Build the full URL
    url = f"{site_domain}/blog/{slug}.html"
    
    # Extract host from domain
    host = site_domain.replace("https://", "").replace("http://", "").strip("/")
    
    return submit_urls([url], host, key, key_location)


def submit_multiple_blog_posts(
    site_domain: str,
    slugs: List[str],
    key: str,
    key_location: Optional[str] = None
) -> bool:
    """
    Submit multiple blog post URLs to IndexNow.
    
    Args:
        site_domain: The site domain (e.g., "https://www.example.com")
        slugs: List of blog post slugs
        key: The IndexNow API key
        key_location: Optional location of the key file
    
    Returns:
        True if submission was successful, False otherwise
    """
    # Ensure site_domain doesn't have trailing slash
    site_domain = site_domain.rstrip("/")
    
    # Build full URLs
    urls = [f"{site_domain}/blog/{slug}.html" for slug in slugs]
    
    # Extract host from domain
    host = site_domain.replace("https://", "").replace("http://", "").strip("/")
    
    return submit_urls(urls, host, key, key_location)

