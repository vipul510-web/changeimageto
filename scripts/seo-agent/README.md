# SEO Agent – Automated GA4 + GSC Insights & Implementation

This agent fetches data from Google Analytics 4 and Google Search Console, combines them to generate SEO tactics, implements safe changes on the site, and emails you a report.

**Two modes:**
- **`run_agent.py`** – Original scripted flow (meta updates, on-page, internal links, email)
- **`run_agentic.py`** – Agentic flow: Observe → Decide → Act → Log, with state store

## What it does

1. **Fetches** – GA4 (traffic, top pages) + GSC (keywords, pages, query+page for cannibalization)
2. **Analyzes** – Six analysis types:
   - **Keyword conflicts** – Detects when pages compete for the same query (cannibalization). Shows which URL should win, which to merge or 301 redirect.
   - **Low-hanging fruit** – Keywords at position 31+ (page 4+) ripe for pushing into top 1–3 with small optimizations.
   - **On-page check** – Verifies top queries appear in title, meta description, H1. Fixes gaps automatically.
   - **Keyword clusters** – Groups related keywords, tracks aggregate performance. Identifies thin clusters for content opportunities.
   - **Competitor gaps** – (Optional, needs SerpAPI) Fetches pages 1–3 (30 results) for target keywords. Shows our position vs top competitors (titles, snippets), identifies where we lag, and auto-adds competitor keywords to our meta when we rank 4–15.
   - **Declining keywords, low-CTR, traffic gaps** – Original tactics.
3. **Implements** – Adds keywords to meta, improves descriptions, fixes on-page gaps (title/meta/H1)
4. **Pushes** – Commits and pushes changes to `origin main` (production)
5. **Emails** – Sends a detailed report with all insights and changes

## Setup

### 1. Install dependencies

```bash
cd /path/to/remback
pip install -r scripts/seo-agent/requirements.txt
# Or use your venv:
.venv-mcp/bin/pip install -r scripts/seo-agent/requirements.txt
```

### 2. Configure

```bash
cp scripts/seo-agent/config.yaml scripts/seo-agent/config.local.yaml
```

Edit `config.local.yaml` – email is pre-filled with vipulagarwal.in@gmail.com. Add your Gmail App Password:

**Option A – In config (less secure):**
```yaml
email:
  smtp_password: "xxxx xxxx xxxx xxxx"  # 16-char App Password
```

**Option B – Via env (recommended for cron):**
```bash
export SEO_AGENT_SMTP_PASSWORD="xxxx xxxx xxxx xxxx"
```

**Get App Password:** [Google Account](https://myaccount.google.com/) → Security → 2-Step Verification → App passwords → Generate.

**Test email:**
```bash
python scripts/seo-agent/run_agent.py --test-email
```

### 3. Run

**Original agent:**
```bash
python scripts/seo-agent/run_agent.py --dry-run
python scripts/seo-agent/run_agent.py
python scripts/seo-agent/run_agent.py --no-email --no-push
```

**Agentic flow (recommended):**
```bash
python scripts/seo-agent/run_agentic.py --dry-run   # Observe + decide only
python scripts/seo-agent/run_agentic.py             # Full run with state logging
python scripts/seo-agent/run_agentic.py --no-email --no-push
```

The agentic flow saves state to `seo_agent_state.db` (runs, actions, metrics snapshots) for future feedback loops.

## Automation (cron)

**Every 4 days** (recommended – uses `run_scheduled.py` to skip if run recently):

```bash
crontab -e
# Add (set SEO_AGENT_SMTP_PASSWORD in crontab or your shell profile):
0 9 * * * cd /Users/vipulagarwal/Documents/remback && SEO_AGENT_SMTP_PASSWORD="your-app-password" /Users/vipulagarwal/Documents/remback/.venv-mcp/bin/python scripts/seo-agent/run_scheduled.py
```

Cron runs daily at 9am; `run_scheduled.py` executes `run_agent.py` only if 4+ days have passed since the last run.

**Weekly** (original):

```bash
0 9 * * 1 cd /path/to/remback && python scripts/seo-agent/run_agent.py
```

**GitHub Actions** (runs from GitHub, no local machine needed):

The workflow `.github/workflows/seo-agent.yml` runs every 4 days at 9am UTC. Add these secrets in **Settings → Secrets and variables → Actions**:

| Secret | Required | Description |
|--------|----------|-------------|
| `GA4_CREDENTIALS_JSON` | Yes | Full content of GA4 service account JSON |
| `GSC_CREDENTIALS_JSON` | Yes | Full content of GSC service account JSON |
| `SEO_AGENT_SMTP_USER` | Yes | Gmail address |
| `SEO_AGENT_SMTP_PASSWORD` | Yes | Gmail App Password |
| `SEO_AGENT_TO_EMAIL` | Yes | Where to send the report |
| `SEO_AGENT_FROM_EMAIL` | No | Defaults to smtp_user |
| `GA4_PROPERTY_ID` | No | Defaults from config |
| `GSC_SITE_URL` | No | Defaults from config |
| `SERP_API_KEY` | No | For competitor analysis |
| `GEMINI_API_KEY` | No | For blog creation |
| `BLOG_CREATION_ENABLED` | No | Set to `true` to enable |

You can also trigger a run manually via **Actions → SEO Agent → Run workflow**.

## Safety limits

- `max_meta_updates: 5` – Max meta changes for declining/low-CTR keywords
- `max_low_hanging_fruit: 5` – Max meta changes for position 31+ keywords (separate quota)
- `max_internal_links_added: 3` – Max internal links per run
- `max_onpage_fixes: 5` – Max title/meta/H1 fixes per run
- Test pages are excluded from edits

**Note:** Conflicts and clusters are report-only (merge/redirect and content ideas require manual decisions).

## Competitor analysis (optional)

Fetches SERP pages 1–3 (30 results) for your top keywords. For each keyword:
- **Our position** – Where we rank (or "Not in top 30")
- **Top 3 competitors** – Domain, title, snippet
- **Recommendation** – "On page 1" / "Improve to push to page 1" / "Create blog post"
- **Auto-tactics** – When we rank 4–15, extracts keywords from top competitor titles and adds them to our meta

Add a [SerpAPI](https://serpapi.com) key:

```yaml
# config.local.yaml
serp_api_key: "your-key"
max_competitor_keywords: 12   # Limits API calls
max_competitor_meta_updates: 2
```

## Blog creation (framework-based)

Blog posts are created by a **separate script** using the AI Blog Framework (`ai-blog-structure.jsx`):

```bash
python scripts/seo-agent/run_blog_creator.py
python scripts/seo-agent/run_blog_creator.py --dry-run
python scripts/seo-agent/run_blog_creator.py --max-posts 3
```

**Flow:**
1. **Keywords** – GSC keywords + competitor gaps (SERP) + SERP "Related searches"
2. **Clustering** – Related keywords merged into topics (e.g. "remove background from image" + "remove background from photo" → one topic)
3. **Framework** – Each topic mapped to a content type (How-To Tutorial, Best-of Listicle, X vs Y, FAQ, etc.) from the 5-pillar framework
4. **Generation** – Gemini creates structured posts following the framework's template and section structure

**Config:**
```yaml
# config.local.yaml
gemini_api_key: "your-key"  # Required for blog creation
serp_api_key: "your-key"   # Optional – for competitor + SERP related keywords
blog_min_impressions: 30
max_blog_gaps_reported: 10
max_blog_posts_per_run: 2
```

## Extending

- Add more keyword→page mappings in `analyzer.py` (`KEYWORD_TO_PAGE`)
- Add new tactic types in `analyzer.py` and `implementer.py`
- Tune priorities and thresholds in the analyzer
- Adjust cluster stopwords in `analyzer.py` (`STOPWORDS`) for better grouping
