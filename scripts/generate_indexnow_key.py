#!/usr/bin/env python3
"""
Script to generate an IndexNow API key and create the key file.

Usage:
    python scripts/generate_indexnow_key.py

This will:
1. Generate a random IndexNow API key
2. Create the key file in the frontend directory
3. Print instructions for setting up environment variables
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import indexnow module
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.indexnow import generate_indexnow_key

def main():
    # Generate the key
    key = generate_indexnow_key()
    
    # Get the frontend directory
    frontend_dir = Path(__file__).parent.parent / "frontend"
    frontend_dir.mkdir(exist_ok=True)
    
    # Create the key file
    key_file = frontend_dir / f"{key}.txt"
    key_file.write_text(key, encoding="utf-8")
    
    print("=" * 60)
    print("IndexNow API Key Generated Successfully!")
    print("=" * 60)
    print(f"\nGenerated Key: {key}")
    print(f"Key file created: {key_file}")
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("\n1. Add the following environment variables to your backend:")
    print(f"   INDEXNOW_KEY={key}")
    print("   INDEXNOW_SITE_DOMAIN=https://your-domain.com")
    print("\n2. The key file has been created in the frontend directory.")
    print("   Make sure it's accessible at: https://your-domain.com/{key}.txt")
    print("\n3. Deploy the key file to your frontend hosting (Vercel, etc.)")
    print("\n4. Verify the key file is accessible by visiting:")
    print(f"   https://your-domain.com/{key}.txt")
    print("\n5. Test the integration by publishing a blog post.")
    print("=" * 60)

if __name__ == "__main__":
    main()

