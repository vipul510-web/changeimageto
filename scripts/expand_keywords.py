#!/usr/bin/env python3
"""
Keyword Expansion Script
Generates keyword variations for programmatic SEO
"""

import csv
import json
from typing import List, Dict

# Seed keywords (your main tools)
TOOLS = [
    'remove background',
    'upscale image',
    'enhance image',
    'blur background',
    'change background',
    'remove text from image',
    'remove people from photo',
    'change color of image',
    'convert image format',
    'bulk resize images'
]

# Platforms
PLATFORMS = [
    'shopify', 'amazon', 'etsy', 'ebay', 'woocommerce', 'bigcommerce',
    'wix', 'squarespace', 'facebook marketplace', 'instagram', 'linkedin',
    'pinterest', 'tiktok', 'printful', 'printify', 'zazzle', 'redbubble',
    'mercari', 'poshmark', 'depop', 'grailed', 'stockx', 'goat'
]

# Industries
INDUSTRIES = [
    'ecommerce', 'real estate', 'fashion', 'jewelry', 'furniture',
    'food and beverage', 'beauty', 'health and wellness', 'electronics',
    'home and garden', 'sports and outdoors', 'automotive', 'art and crafts',
    'photography', 'marketing', 'advertising', 'web design', 'print',
    'publishing', 'manufacturing'
]

# Use Cases
USE_CASES = [
    'product photos', 'portraits', 'logos', 'social media posts', 'print',
    'web', 'email marketing', 'presentations', 'linkedin photos',
    'dating app photos', 'professional headshots', 'product listings',
    'catalog photos', 'website images', 'blog images', 'email campaigns',
    'social media ads', 'print ads', 'business cards', 'flyers'
]

# Colors
COLORS = [
    {'name': 'red', 'hex': '#FF0000'},
    {'name': 'green', 'hex': '#00FF00'},
    {'name': 'blue', 'hex': '#0000FF'},
    {'name': 'yellow', 'hex': '#FFFF00'},
    {'name': 'orange', 'hex': '#FFA500'},
    {'name': 'purple', 'hex': '#800080'},
    {'name': 'pink', 'hex': '#FFC0CB'},
    {'name': 'gray', 'hex': '#808080'},
    {'name': 'navy', 'hex': '#000080'},
    {'name': 'teal', 'hex': '#008080'},
    {'name': 'brown', 'hex': '#A52A2A'},
    {'name': 'maroon', 'hex': '#800000'},
    {'name': 'burgundy', 'hex': '#800020'},
    {'name': 'coral', 'hex': '#FF7F50'},
    {'name': 'mint', 'hex': '#98FF98'},
    {'name': 'lavender', 'hex': '#E6E6FA'},
    {'name': 'turquoise', 'hex': '#40E0D0'},
    {'name': 'gold', 'hex': '#FFD700'},
    {'name': 'silver', 'hex': '#C0C0C0'}
]

# Page type mappings
PAGE_TYPE_MAP = {
    'remove background': 'remove_background',
    'upscale image': 'upscale',
    'enhance image': 'enhance',
    'blur background': 'blur',
    'change background': 'change_background',
    'remove text from image': 'remove_text',
    'remove people from photo': 'remove_people',
    'change color of image': 'change_color',
    'convert image format': 'convert_format',
    'bulk resize images': 'bulk_resize'
}

def generate_functionality_keywords() -> List[Dict]:
    """Generate functionality page keywords"""
    keywords = []
    
    # Platform-specific: [tool] for [platform]
    for tool in TOOLS:
        page_type = PAGE_TYPE_MAP.get(tool, 'remove_background')
        for platform in PLATFORMS[:10]:  # Top 10 platforms first
            keyword = f"{tool} for {platform}"
            keywords.append({
                'keyword': keyword,
                'page_type': 'functionality',
                'category': 'platform_specific',
                'search_volume': 0,  # To be researched
                'difficulty': 0,  # To be researched
                'template_type': 'use_case_template',
                'page_type_category': page_type,
                'variables': json.dumps({
                    'platform': platform.title(),
                    'use_case': 'e-commerce' if 'shopify' in platform or 'amazon' in platform else 'general'
                })
            })
    
    # Industry-specific: [tool] for [industry]
    for tool in TOOLS[:5]:  # Top 5 tools
        page_type = PAGE_TYPE_MAP.get(tool, 'remove_background')
        for industry in INDUSTRIES[:10]:  # Top 10 industries
            keyword = f"{tool} for {industry}"
            keywords.append({
                'keyword': keyword,
                'page_type': 'functionality',
                'category': 'industry_specific',
                'search_volume': 0,
                'difficulty': 0,
                'template_type': 'use_case_template',
                'page_type_category': page_type,
                'variables': json.dumps({
                    'industry': industry,
                    'use_case': industry
                })
            })
    
    # Use case-specific: [tool] for [use case]
    for tool in TOOLS[:5]:
        page_type = PAGE_TYPE_MAP.get(tool, 'remove_background')
        for use_case in USE_CASES[:10]:
            keyword = f"{tool} for {use_case}"
            keywords.append({
                'keyword': keyword,
                'page_type': 'functionality',
                'category': 'use_case_specific',
                'search_volume': 0,
                'difficulty': 0,
                'template_type': 'use_case_template',
                'page_type_category': page_type,
                'variables': json.dumps({
                    'use_case': use_case,
                    'image_type': use_case
                })
            })
    
    # Color-specific: change background to [color]
    for color in COLORS:
        keywords.append({
            'keyword': f"change background to {color['name']}",
            'page_type': 'functionality',
            'category': 'color_specific',
            'search_volume': 0,
            'difficulty': 0,
            'template_type': 'color_template',
            'page_type_category': 'change_background_color',
            'variables': json.dumps({
                'color': color['name'],
                'hex': color['hex'],
                'use_cases': 'product photos, marketing'
            })
        })
    
    return keywords

def generate_blog_keywords() -> List[Dict]:
    """Generate blog page keywords"""
    keywords = []
    
    # Platform tutorials: how to [tool] for [platform]
    for tool in TOOLS[:5]:
        page_type = PAGE_TYPE_MAP.get(tool, 'remove_background')
        for platform in PLATFORMS[:10]:
            keyword = f"how to {tool} for {platform}"
            keywords.append({
                'keyword': keyword,
                'page_type': 'blog',
                'category': 'platform_tutorial',
                'search_volume': 0,
                'difficulty': 0,
                'template_type': 'tutorial_template',
                'page_type_category': page_type,
                'variables': json.dumps({
                    'platform': platform.title(),
                    'use_case': 'e-commerce',
                    'action': tool
                })
            })
    
    # Comparison pages: best [tool] for [platform]
    for tool in TOOLS[:5]:
        page_type = PAGE_TYPE_MAP.get(tool, 'remove_background')
        for platform in PLATFORMS[:5]:  # Top 5 platforms
            keyword = f"best {tool} for {platform}"
            keywords.append({
                'keyword': keyword,
                'page_type': 'blog',
                'category': 'comparison',
                'search_volume': 0,
                'difficulty': 0,
                'template_type': 'comparison_template',
                'page_type_category': page_type,
                'variables': json.dumps({
                    'platform': platform.title(),
                    'tool_a': 'ChangeImageTo',
                    'tool_b': 'Competitor'
                })
            })
    
    return keywords

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate keyword variations')
    parser.add_argument('--output', default='expanded_keywords.csv',
                       help='Output CSV file')
    parser.add_argument('--functionality-only', action='store_true',
                       help='Only generate functionality keywords')
    parser.add_argument('--blog-only', action='store_true',
                       help='Only generate blog keywords')
    
    args = parser.parse_args()
    
    all_keywords = []
    
    if not args.blog_only:
        print("Generating functionality keywords...")
        func_keywords = generate_functionality_keywords()
        all_keywords.extend(func_keywords)
        print(f"  Generated {len(func_keywords)} functionality keywords")
    
    if not args.functionality_only:
        print("Generating blog keywords...")
        blog_keywords = generate_blog_keywords()
        all_keywords.extend(blog_keywords)
        print(f"  Generated {len(blog_keywords)} blog keywords")
    
    # Write to CSV
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(args.output) == args.output:  # Just filename, no path
        output_path = os.path.join(script_dir, args.output)
    else:
        output_path = args.output
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'keyword', 'page_type', 'category', 'search_volume', 'difficulty',
            'template_type', 'page_type_category', 'variables'
        ])
        writer.writeheader()
        writer.writerows(all_keywords)
    
    print(f"\n✅ Generated {len(all_keywords)} keywords")
    print(f"📝 Saved to: {output_path}")
    print("\nNext steps:")
    print("1. Research search volumes using Google Keyword Planner")
    print("2. Update search_volume and difficulty columns")
    print("3. Review and filter keywords")
    print("4. Merge with existing programmatic_seo_keywords.csv")
    print("5. Generate pages using generate_functionality_pages.py or generate_programmatic_seo.py")

if __name__ == '__main__':
    main()

