#!/usr/bin/env python3
"""
SEO Agent – Automated GA4 + GSC analysis and implementation.

Run: python scripts/seo-agent/run_agent.py
Dry run (no edits, no email): python scripts/seo-agent/run_agent.py --dry-run

Setup:
1. Copy config.yaml to config.local.yaml
2. Fill in email credentials (Gmail App Password recommended)
3. cron: 0 9 * * 1 cd /path/to/remback && python scripts/seo-agent/run_agent.py
   (weekly Monday 9am)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add agent dir for imports, run from project root
AGENT_DIR = Path(__file__).parent
PROJECT_ROOT = AGENT_DIR.parent.parent
sys.path.insert(0, str(AGENT_DIR))
os.chdir(PROJECT_ROOT)


def git_commit_and_push(changed_files: list[str], project_root: Path) -> tuple[bool, str]:
    """
    Commit and push changed files to production.
    Returns (success, message).
    """
    if not changed_files:
        return True, "No files to commit"

    # Convert to paths relative to project root
    rel_paths = []
    for f in changed_files:
        p = Path(f)
        try:
            rel = p.relative_to(project_root)
            rel_paths.append(str(rel))
        except ValueError:
            rel_paths.append(f)

    try:
        subprocess.run(["git", "add"] + rel_paths, check=True, capture_output=True, cwd=project_root)
        msg = "SEO Agent: meta keyword updates and improvements"
        commit = subprocess.run(
            ["git", "commit", "-m", msg],
            capture_output=True,
            cwd=project_root,
        )
        if commit.returncode != 0:
            if b"nothing to commit" in commit.stdout or b"nothing to commit" in (commit.stderr or b""):
                return True, "No new changes to commit (already up to date)"
            return False, (commit.stderr or commit.stdout).decode()
        push = subprocess.run(["git", "push", "origin", "main"], capture_output=True, cwd=project_root)
        if push.returncode != 0:
            return False, (push.stderr or push.stdout).decode()
        return True, f"Pushed {len(rel_paths)} file(s) to production"
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode() if e.stderr else str(e)
        return False, f"Git error: {err}"


def main():
    parser = argparse.ArgumentParser(description="SEO Agent")
    parser.add_argument("--dry-run", action="store_true", help="No edits, no email, no push")
    parser.add_argument("--no-email", action="store_true", help="Skip email even if configured")
    parser.add_argument("--no-push", action="store_true", help="Skip git commit and push")
    parser.add_argument("--test-email", action="store_true", help="Send a test email and exit")
    args = parser.parse_args()

    from data_fetcher import fetch_ga4_data, fetch_gsc_data, fetch_gsc_comparison, fetch_gsc_query_page_data, load_config
    from analyzer import generate_tactics, select_implementable_tactics
    from implementer import implement_tactic
    from email_reporter import send_report

    config = load_config()
    print("SEO Agent – ChangeImageTo.com")
    print("=" * 40)

    if args.test_email:
        from email_reporter import send_report
        ok = send_report([], [{"tactic": "Test", "success": True, "message": "Test email", "file_changed": None}], config, dry_run=False, insights={})
        print("Test email sent!" if ok else "Test email failed. Check config and SEO_AGENT_SMTP_PASSWORD.")
        return 0 if ok else 1

    # 1. Fetch data
    print("Fetching GA4...")
    ga4 = fetch_ga4_data(config)
    print(f"  Top pages: {len(ga4['pages'])}")

    print("Fetching GSC...")
    gsc = fetch_gsc_data(config)
    print(f"  Keywords: {len(gsc['keywords'])}, Pages: {len(gsc['pages'])}")

    print("Fetching GSC query+page (cannibalization)...")
    gsc_query_page = fetch_gsc_query_page_data(config)
    print(f"  Query-page rows: {len(gsc_query_page['query_page'])}")

    print("Fetching GSC decliners...")
    decliners = fetch_gsc_comparison(config)
    print(f"  Declining keywords: {len(decliners)}")

    # 2. Generate tactics and insights
    result = generate_tactics(ga4, gsc, decliners, gsc_query_page, config)
    tactics = result["tactics"]
    insights = result["insights"]
    implementable = select_implementable_tactics(tactics, config)
    print(f"\nTactics: {len(tactics)} total, {len(implementable)} implementable")
    print(f"Insights: {len(insights['conflicts'])} conflicts, {len(insights['low_hanging'])} low-hanging, {len(insights['onpage_gaps'])} on-page gaps, {len(insights['clusters'])} clusters, {len(insights['competitor_gaps'])} competitor gaps")

    frontend = config.get("frontend_dir", "frontend")

    # 3. Implement
    implemented = []

    if args.dry_run:
        print("\n[DRY RUN] Would implement:")
        for t in implementable:
            print(f"  - {t.get('description')}")
    else:
        for t in implementable:
            result = implement_tactic(t, frontend)
            implemented.append({
                "tactic": t.get("description"),
                "success": result["success"],
                "message": result["message"],
                "file_changed": result.get("file_changed"),
            })
            status = "✓" if result["success"] else "✗"
            print(f"  {status} {t.get('description')} -> {result['message']}")

    # 4. Git commit and push (only if changes were made)
    if not args.dry_run and not args.no_push and implemented:
        changed_files = []
        for i in implemented:
            if i.get("success") and i.get("file_changed"):
                fc = i["file_changed"]
                changed_files.extend(fc if isinstance(fc, list) else [fc])
        if changed_files:
            success, msg = git_commit_and_push(changed_files, PROJECT_ROOT)
            if success:
                print(f"\n{msg}")
            else:
                print(f"\nGit push failed: {msg}")

    # 5. Email
    if not args.dry_run and not args.no_email:
        if send_report(tactics, implemented, config, dry_run=False, insights=insights):
            print("\nEmail sent.")
        else:
            print("\nEmail not sent (check config).")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
