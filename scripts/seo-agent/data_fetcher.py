"""
Fetch data from Google Analytics 4 and Google Search Console.
"""
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)


def load_config():
    """Load config from config.local.yaml or config.yaml"""
    import yaml
    local = Path(__file__).parent / "config.local.yaml"
    default = Path(__file__).parent / "config.yaml"
    path = local if local.exists() else default
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_ga4_data(config, days=30):
    """Fetch GA4 traffic and page data."""
    from google.oauth2.service_account import Credentials
    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.analytics.data_v1beta.types import (
        DateRange,
        Dimension,
        Metric,
        RunReportRequest,
    )

    creds_path = PROJECT_ROOT / config["ga4_credentials"]
    creds = Credentials.from_service_account_file(str(creds_path))
    client = BetaAnalyticsDataClient(credentials=creds)

    end = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days + 3)).strftime("%Y-%m-%d")

    # Top pages by traffic
    response = client.run_report(
        RunReportRequest(
            property=f"properties/{config['ga4_property_id']}",
            dimensions=[Dimension(name="pagePath")],
            metrics=[
                Metric(name="screenPageViews"),
                Metric(name="sessions"),
                Metric(name="totalUsers"),
                Metric(name="bounceRate"),
                Metric(name="averageSessionDuration"),
            ],
            date_ranges=[DateRange(start_date=start, end_date=end)],
            limit=100,
        )
    )

    pages = []
    for row in response.rows:
        path = row.dimension_values[0].value
        pages.append({
            "path": path,
            "page_views": int(row.metric_values[0].value),
            "sessions": int(row.metric_values[1].value),
            "users": int(row.metric_values[2].value),
            "bounce_rate": float(row.metric_values[3].value) * 100 if row.metric_values[3].value else 0,
            "avg_session_duration": float(row.metric_values[4].value) if row.metric_values[4].value else 0,
        })
    return {"pages": pages, "date_range": f"{start} to {end}"}


def fetch_gsc_data(config, days=60):
    """Fetch GSC search analytics - keywords and pages."""
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    creds_path = PROJECT_ROOT / config["gsc_credentials"]
    creds = Credentials.from_service_account_file(str(creds_path))
    service = build("searchconsole", "v1", credentials=creds)

    end = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Keywords (queries)
    query_resp = service.searchanalytics().query(
        siteUrl=config["gsc_site_url"],
        body={
            "startDate": start,
            "endDate": end,
            "dimensions": ["query"],
            "rowLimit": 500,
        },
    ).execute()

    keywords = []
    for row in query_resp.get("rows", []):
        keywords.append({
            "query": row["keys"][0],
            "clicks": row.get("clicks", 0),
            "impressions": row.get("impressions", 0),
            "ctr": round(row.get("ctr", 0) * 100, 2),
            "position": round(row.get("position", 0), 1),
        })

    # Pages
    page_resp = service.searchanalytics().query(
        siteUrl=config["gsc_site_url"],
        body={
            "startDate": start,
            "endDate": end,
            "dimensions": ["page"],
            "rowLimit": 200,
        },
    ).execute()

    pages = []
    for row in page_resp.get("rows", []):
        url = row["keys"][0]
        path = url.replace("https://www.changeimageto.com", "").replace("https://changeimageto.com", "")
        if not path:
            path = "/"
        pages.append({
            "path": path,
            "url": url,
            "clicks": row.get("clicks", 0),
            "impressions": row.get("impressions", 0),
            "ctr": round(row.get("ctr", 0) * 100, 2),
            "position": round(row.get("position", 0), 1),
        })

    return {
        "keywords": keywords,
        "pages": pages,
        "date_range": f"{start} to {end}",
    }


def fetch_gsc_comparison(config):
    """Fetch GSC data for current vs previous 30 days to find decliners."""
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    creds_path = PROJECT_ROOT / config["gsc_credentials"]
    creds = Credentials.from_service_account_file(str(creds_path))
    service = build("searchconsole", "v1", credentials=creds)

    today = datetime.now()
    curr_end = (today - timedelta(days=3)).strftime("%Y-%m-%d")
    curr_start = (today - timedelta(days=33)).strftime("%Y-%m-%d")
    prev_end = (today - timedelta(days=34)).strftime("%Y-%m-%d")
    prev_start = (today - timedelta(days=64)).strftime("%Y-%m-%d")

    curr = service.searchanalytics().query(
        siteUrl=config["gsc_site_url"],
        body={"startDate": curr_start, "endDate": curr_end, "dimensions": ["query"], "rowLimit": 500},
    ).execute()

    prev = service.searchanalytics().query(
        siteUrl=config["gsc_site_url"],
        body={"startDate": prev_start, "endDate": prev_end, "dimensions": ["query"], "rowLimit": 500},
    ).execute()

    prev_data = {
        row["keys"][0]: {
            "clicks": row.get("clicks", 0),
            "impressions": row.get("impressions", 0),
            "position": row.get("position", 0),
        }
        for row in prev.get("rows", [])
    }

    decliners = []
    for row in curr.get("rows", []):
        q = row["keys"][0]
        curr_c, curr_i, curr_p = row.get("clicks", 0), row.get("impressions", 0), row.get("position", 0)
        prev_info = prev_data.get(q, {})
        prev_c, prev_i, prev_p = prev_info.get("clicks", 0), prev_info.get("impressions", 0), prev_info.get("position", 0)
        imp_drop = prev_i - curr_i
        click_drop = prev_c - curr_c
        if imp_drop > 5 or click_drop > 0:
            decliners.append({
                "query": q,
                "prev_clicks": prev_c, "curr_clicks": curr_c, "click_drop": click_drop,
                "prev_impressions": prev_i, "curr_impressions": curr_i, "imp_drop": imp_drop,
                "prev_position": round(prev_p, 1), "curr_position": round(curr_p, 1),
            })

    decliners.sort(key=lambda x: (x["click_drop"], x["imp_drop"]), reverse=True)
    return decliners


def fetch_gsc_query_page_data(config, days=60):
    """
    Fetch GSC data with query + page dimensions for cannibalization detection.
    Returns list of {query, page, path, clicks, impressions, ctr, position}.
    """
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    creds_path = PROJECT_ROOT / config["gsc_credentials"]
    creds = Credentials.from_service_account_file(str(creds_path))
    service = build("searchconsole", "v1", credentials=creds)

    end = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    resp = service.searchanalytics().query(
        siteUrl=config["gsc_site_url"],
        body={
            "startDate": start,
            "endDate": end,
            "dimensions": ["query", "page"],
            "rowLimit": 25000,
        },
    ).execute()

    rows = []
    for row in resp.get("rows", []):
        url = row["keys"][1]
        path = url.replace("https://www.changeimageto.com", "").replace("https://changeimageto.com", "")
        if not path:
            path = "/"
        rows.append({
            "query": row["keys"][0],
            "page": url,
            "path": path,
            "clicks": row.get("clicks", 0),
            "impressions": row.get("impressions", 0),
            "ctr": round(row.get("ctr", 0) * 100, 2),
            "position": round(row.get("position", 0), 1),
        })
    return {"query_page": rows, "date_range": f"{start} to {end}"}


if __name__ == "__main__":
    config = load_config()
    print("Fetching GA4...")
    ga4 = fetch_ga4_data(config)
    print(f"  Top pages: {len(ga4['pages'])}")
    print("Fetching GSC...")
    gsc = fetch_gsc_data(config)
    print(f"  Keywords: {len(gsc['keywords'])}, Pages: {len(gsc['pages'])}")
    print("Fetching GSC decliners...")
    decliners = fetch_gsc_comparison(config)
    print(f"  Declining keywords: {len(decliners)}")
