#!/usr/bin/env python3
"""Add Google AdSense script to all HTML files that don't have it."""

import os
from pathlib import Path

ADSENSE_SNIPPET = '''
<!-- Google AdSense - Replace ca-pub-4059286158672130 with your publisher ID from adsense.google.com -->
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-4059286158672130"
     crossorigin="anonymous"></script>
'''

def add_adsense_to_file(filepath: Path) -> bool:
    """Add AdSense script before </head> if not already present. Returns True if modified."""
    content = filepath.read_text(encoding='utf-8', errors='replace')
    if 'adsbygoogle' in content or 'adsense' in content.lower():
        return False
    if '</head>' not in content:
        return False
    # Insert before </head>
    new_content = content.replace('</head>', ADSENSE_SNIPPET + '\n  </head>')
    if new_content == content:
        return False
    filepath.write_text(new_content, encoding='utf-8')
    return True

def main():
    frontend = Path(__file__).parent / 'frontend'
    modified = []
    for html in frontend.rglob('*.html'):
        if add_adsense_to_file(html):
            modified.append(str(html.relative_to(frontend)))
    if modified:
        print(f"Added AdSense to {len(modified)} files:")
        for m in sorted(modified):
            print(f"  - {m}")
    else:
        print("No files needed updating (AdSense already present or no </head> found).")

if __name__ == '__main__':
    main()
