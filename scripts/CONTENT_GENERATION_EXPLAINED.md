# Content Generation Explained

## How Content is Created

### For Functionality Pages (Tool Pages)

**Current Method: Template-Based Generation**

The content for functionality pages like `remove-background-for-shopify.html` is created using **template-based generation** (not OpenAI).

#### How It Works:

1. **Base Template**: Copies an existing functionality page (e.g., `remove-background-from-image.html`)
2. **Customization**: Updates title, meta tags, H1, and SEO content section
3. **SEO Content**: Generates SEO content using a template function that fills in variables

#### The Tool: `generate_functionality_pages.py`

```bash
cd scripts
python3 generate_functionality_pages.py --limit 5
```

**What it does:**
- Reads keywords from CSV file
- Copies base template (e.g., `remove-background-from-image.html`)
- Customizes:
  - Title tag
  - Meta description
  - H1 heading
  - SEO content section (with platform/industry-specific content)
- Saves as new HTML file

**SEO Content Template:**
The script uses `generate_seo_content()` function which creates:
- Introduction paragraph
- "Why [keyword]?" section (platform/industry-specific)
- Step-by-step guide
- Tips for best results
- FAQ section (with schema markup)
- Related tools links

**Example:**
```python
# For "remove background for shopify"
# Variables: {"platform": "Shopify", "use_case": "e-commerce"}
# Generates content like:
# - "Shopify Listings: Professional product photos..."
# - "Fast Processing: Perfect for bulk editing Shopify product photos..."
```

### For Blog Pages (Content Pages)

**Two Options:**

#### Option 1: Template-Based (Default)
```bash
python3 generate_programmatic_seo.py --limit 10
```
- Uses simple templates
- Fast and free
- Less natural language

#### Option 2: OpenAI-Generated (Better Quality)
```bash
export OPENAI_API_KEY="sk-..."
python3 generate_programmatic_seo.py --use-openai --limit 10
```
- Uses GPT-4o-mini or GPT-4o
- Natural, unique content
- Costs ~$0.01-0.05 per article

## Your Current Page

The `remove-background-for-shopify.html` page you created appears to have been:
1. **Manually created** - You copied a base template
2. **Template-based content** - The SEO section matches our template format

## For Future Pages

### Automated Generation (Recommended)

Instead of creating pages manually, use the automated tool:

```bash
# 1. Add keyword to CSV
# Edit: scripts/programmatic_seo_keywords.csv
# Add: remove background for amazon,functionality,use_case_specific,600,28,use_case_template,remove_background,"{""platform"": ""Amazon""}"

# 2. Generate the page
cd scripts
python3 generate_functionality_pages.py --limit 1

# 3. Review the generated file
# Check: frontend/remove-background-for-amazon.html

# 4. Commit and push
git add frontend/remove-background-for-amazon.html frontend/sitemap.xml
git commit -m "Add remove-background-for-amazon.html"
git push origin main
```

### Manual Creation (If Needed)

If you need to manually create a page:
1. Copy an existing functionality page (e.g., `remove-background-from-image.html`)
2. Update:
   - Title tag
   - Meta description
   - H1 heading
   - Header description
   - SEO content section (or use the template format)
3. Save with new filename

## Content Quality

### Template-Based (Current for Functionality Pages)
- ✅ Consistent structure
- ✅ Fast generation
- ✅ Free
- ❌ Less unique
- ❌ Can be repetitive

### OpenAI-Generated (Available for Blog Pages)
- ✅ Natural, unique content
- ✅ Better SEO
- ✅ More engaging
- ❌ Costs money
- ❌ Slower

## Recommendation

**For Functionality Pages:**
- Use template-based generation (current approach)
- Fast, free, consistent
- Good enough for tool pages

**For Blog Pages:**
- Use OpenAI for high-priority keywords (500+ volume)
- Use templates for bulk generation
- Hybrid approach works best

## Next Steps

1. **Use automated tool** for future functionality pages
2. **Add keywords to CSV** instead of creating manually
3. **Batch generate** multiple pages at once
4. **Review and optimize** generated content before publishing

