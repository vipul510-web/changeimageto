#!/usr/bin/env python3
"""
Programmatic SEO Content Generator
Generates blog posts from keyword data and templates
"""

import csv
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import os

# Database connection
DB_PATH = 'blog_management.db'

def init_db():
    """Initialize database with programmatic SEO columns"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS blog_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'draft',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            published_at TIMESTAMP NULL,
            author TEXT DEFAULT 'system',
            notes TEXT,
            metadata TEXT,
            keyword TEXT,
            keyword_category TEXT,
            search_volume INTEGER,
            difficulty_score REAL,
            template_type TEXT,
            variables TEXT,
            page_type TEXT DEFAULT 'blog',
            generated_at TIMESTAMP,
            last_updated TIMESTAMP
        )
    ''')
    
    # Add columns if they don't exist (for existing databases)
    columns_to_add = [
        ('keyword', 'TEXT'),
        ('keyword_category', 'TEXT'),
        ('search_volume', 'INTEGER'),
        ('difficulty_score', 'REAL'),
        ('template_type', 'TEXT'),
        ('variables', 'TEXT'),
        ('page_type', 'TEXT DEFAULT "blog"'),
        ('generated_at', 'TIMESTAMP'),
        ('last_updated', 'TIMESTAMP')
    ]
    
    # Get existing columns
    cursor.execute("PRAGMA table_info(blog_posts)")
    existing_columns = [row[1] for row in cursor.fetchall()]
    
    # Add missing columns
    for col_name, col_type in columns_to_add:
        if col_name not in existing_columns:
            try:
                cursor.execute(f'ALTER TABLE blog_posts ADD COLUMN {col_name} {col_type}')
            except sqlite3.OperationalError as e:
                print(f"Warning: Could not add column {col_name}: {e}")
    
    conn.commit()
    conn.close()

def normalize_slug(keyword: str) -> str:
    """Convert keyword to URL-friendly slug"""
    slug = keyword.lower().strip()
    slug = slug.replace(' ', '-')
    slug = slug.replace('_', '-')
    # Remove special characters
    slug = ''.join(c if c.isalnum() or c == '-' else '' for c in slug)
    # Remove multiple dashes
    while '--' in slug:
        slug = slug.replace('--', '-')
    return slug.strip('-')

def generate_content_from_template(keyword: str, category: str, template_type: str, variables: Dict) -> str:
    """Generate blog post content from template"""
    
    # Simple template-based generation
    # In production, you'd use more sophisticated templates or LLM
    
    tool = variables.get('tool', 'ChangeImageTo')
    use_case = variables.get('use_case', 'image editing')
    color = variables.get('color', '')
    
    content = f"""# {keyword.title()}

## Introduction

{keyword.title()} is a common task for many users. Whether you're working with {use_case}, having the right tools and methods can make all the difference. In this comprehensive guide, we'll show you multiple ways to {keyword}, including the easiest online method using ChangeImageTo.

## Why {keyword.title()}?

{keyword.title()} is essential for various purposes:
- Professional image editing
- Creating clean product photos
- Enhancing visual content
- Preparing images for presentations

## Method 1: Using ChangeImageTo (Recommended)

The easiest way to {keyword} is using ChangeImageTo, our free online tool:

### Step-by-Step Guide:

1. **Visit ChangeImageTo.com**
   - Go to https://www.changeimageto.com
   - No signup required

2. **Upload Your Image**
   - Click "Upload Image" or drag and drop
   - Supports JPG, PNG, WebP formats

3. **Process Your Image**
   - Select your desired action
   - Wait a few seconds for processing

4. **Download Your Result**
   - Preview the result
   - Click download to save

### Advantages of ChangeImageTo:
- ✅ Free to use
- ✅ No signup required
- ✅ Fast processing
- ✅ High-quality results
- ✅ Works on any device

## Method 2: Using {tool}

If you prefer using {tool}, here's how:

[Tool-specific instructions would go here]

## Method 3: Manual Method

[Manual method instructions if applicable]

## Tips and Best Practices

1. **Image Quality**: Start with high-resolution images for best results
2. **File Format**: PNG format preserves transparency
3. **Background Choice**: Consider your use case when choosing background colors
4. **Batch Processing**: Use ChangeImageTo for processing multiple images

## Frequently Asked Questions

### Q: Is ChangeImageTo free?
A: Yes, ChangeImageTo is completely free to use with no signup required.

### Q: What image formats are supported?
A: ChangeImageTo supports JPG, PNG, and WebP formats.

### Q: How long does processing take?
A: Most images are processed in 2-5 seconds.

### Q: Can I use this for commercial purposes?
A: Yes, you can use processed images for any purpose.

## Conclusion

{keyword.title()} doesn't have to be complicated. With tools like ChangeImageTo, you can achieve professional results in seconds without any technical knowledge. Try it free today at https://www.changeimageto.com!

## Related Tools and Resources

- [Change Background Color](/change-image-background.html)
- [Upscale Image](/upscale-image.html)
- [Enhance Image](/enhance-image.html)
- [Remove Background](/remove-background-from-image.html)
"""
    
    return content

def import_keywords_from_csv(csv_path: str) -> List[Dict]:
    """Import keywords from CSV file"""
    keywords = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse variables JSON
            variables = {}
            if row.get('variables'):
                try:
                    variables = json.loads(row['variables'])
                except json.JSONDecodeError:
                    pass
            
            keywords.append({
                'keyword': row['keyword'],
                'page_type': row.get('page_type', 'blog'),  # 'blog' or 'functionality'
                'category': row['category'],
                'search_volume': int(row.get('search_volume', 0)),
                'difficulty': float(row.get('difficulty', 0)),
                'template_type': row.get('template_type', 'default'),
                'page_type_category': row.get('page_type_category', ''),  # For functionality pages
                'variables': variables
            })
    
    return keywords

def create_blog_post(keyword_data: Dict, content: str) -> Optional[int]:
    """Create blog post in database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    slug = normalize_slug(keyword_data['keyword'])
    title = keyword_data['keyword'].title()
    
    # Check if slug already exists
    cursor.execute('SELECT id FROM blog_posts WHERE slug = ?', (slug,))
    if cursor.fetchone():
        print(f"⚠️  Slug already exists: {slug}")
        conn.close()
        return None
    
    # Insert new post
    cursor.execute('''
        INSERT INTO blog_posts 
        (slug, title, content, status, keyword, keyword_category, search_volume, 
         difficulty_score, template_type, variables, created_at)
        VALUES (?, ?, ?, 'draft', ?, ?, ?, ?, ?, ?, ?)
    ''', (
        slug,
        title,
        content,
        keyword_data['keyword'],
        keyword_data['category'],
        keyword_data['search_volume'],
        keyword_data['difficulty'],
        keyword_data['template_type'],
        json.dumps(keyword_data['variables']),
        datetime.utcnow().isoformat()
    ))
    
    post_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    print(f"✅ Created draft: {slug} (ID: {post_id})")
    return post_id

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate programmatic SEO blog posts')
    parser.add_argument('--csv', default='programmatic_seo_keywords.csv', 
                       help='CSV file with keywords')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of posts to generate')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be created without actually creating')
    parser.add_argument('--blog-only', action='store_true',
                       help='Only generate blog pages (skip functionality pages)')
    
    args = parser.parse_args()
    
    # Initialize database
    init_db()
    
    # Import keywords
    csv_path = os.path.join(os.path.dirname(__file__), args.csv)
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    keywords = import_keywords_from_csv(csv_path)
    
    # Filter by page type
    if args.blog_only:
        keywords = [k for k in keywords if k.get('page_type') == 'blog']
    else:
        # Only process blog pages in this script (functionality pages use separate script)
        keywords = [k for k in keywords if k.get('page_type') == 'blog']
    
    if args.limit:
        keywords = keywords[:args.limit]
    
    print(f"📝 Generating {len(keywords)} blog posts...")
    print()
    
    created = 0
    skipped = 0
    
    for kw_data in keywords:
        slug = normalize_slug(kw_data['keyword'])
        
        if args.dry_run:
            print(f"Would create: {slug}")
            print(f"  Keyword: {kw_data['keyword']}")
            print(f"  Category: {kw_data['category']}")
            print(f"  Search Volume: {kw_data['search_volume']}")
            print()
            continue
        
        # Generate content
        content = generate_content_from_template(
            kw_data['keyword'],
            kw_data['category'],
            kw_data['template_type'],
            kw_data['variables']
        )
        
        # Create blog post
        post_id = create_blog_post(kw_data, content)
        
        if post_id:
            created += 1
        else:
            skipped += 1
    
    print()
    print(f"✅ Created: {created}")
    print(f"⚠️  Skipped: {skipped}")
    print()
    print("Next steps:")
    print("1. Review drafts in blog admin panel")
    print("2. Approve and publish high-quality posts")
    print("3. Monitor performance in Google Search Console")
    print()
    print("Note: Use generate_functionality_pages.py for functionality pages")

if __name__ == '__main__':
    main()

