#!/usr/bin/env python3
"""
Create config.local.yaml and credential files from environment variables.
Used by GitHub Actions / CI – do not run locally (overwrites config).
"""
import os
from pathlib import Path

AGENT_DIR = Path(__file__).parent
PROJECT_ROOT = AGENT_DIR.parent.parent


def main():
    import json
    ga4_json = (os.environ.get("GA4_CREDENTIALS_JSON") or "").strip()
    gsc_json = (os.environ.get("GSC_CREDENTIALS_JSON") or "").strip()
    if not ga4_json:
        raise SystemExit("ERROR: GA4_CREDENTIALS_JSON secret is empty or not set. Add it in Settings → Secrets.")
    if not gsc_json:
        raise SystemExit("ERROR: GSC_CREDENTIALS_JSON secret is empty or not set. Add it in Settings → Secrets.")

    try:
        json.loads(ga4_json)
        json.loads(gsc_json)
    except json.JSONDecodeError as e:
        raise SystemExit(f"ERROR: Invalid JSON in credentials: {e}")

    root = PROJECT_ROOT.resolve()
    ga4_path = root / "gemini-test-487909-7e2ee8971cff.json"
    gsc_path = root / "gemini-test-487909-c351b5b6a299.json"
    ga4_path.write_text(ga4_json)
    gsc_path.write_text(gsc_json)

    import yaml
    config = {
        "ga4_credentials": "gemini-test-487909-7e2ee8971cff.json",
        "gsc_credentials": "gemini-test-487909-c351b5b6a299.json",
        "ga4_property_id": os.environ.get("GA4_PROPERTY_ID", "505035310"),
        "gsc_site_url": os.environ.get("GSC_SITE_URL", "sc-domain:changeimageto.com"),
        "frontend_dir": "frontend",
        "project_root": ".",
        "email": {
            "enabled": True,
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": os.environ.get("SEO_AGENT_SMTP_USER", ""),
            "smtp_password": os.environ.get("SEO_AGENT_SMTP_PASSWORD", ""),
            "from_email": os.environ.get("SEO_AGENT_FROM_EMAIL", ""),
            "to_email": os.environ.get("SEO_AGENT_TO_EMAIL", ""),
        },
        "max_meta_updates": 5,
        "max_low_hanging_fruit": 5,
        "max_internal_links_added": 3,
        "max_onpage_fixes": 5,
        "exclude_patterns": ["test-", "blog-admin", "feedback-popup"],
    }
    if os.environ.get("SERP_API_KEY"):
        config["serp_api_key"] = os.environ["SERP_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        config["gemini_api_key"] = os.environ["GEMINI_API_KEY"]
        config["blog_creation_enabled"] = os.environ.get("BLOG_CREATION_ENABLED", "false").lower() == "true"

    config_path = AGENT_DIR / "config.local.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Created {config_path}")


if __name__ == "__main__":
    main()
