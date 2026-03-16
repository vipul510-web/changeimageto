"""
Send SEO agent report via email.
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path


def send_report(
    tactics: list,
    implemented: list,
    config: dict,
    dry_run: bool = False,
    insights: dict | None = None,
) -> bool:
    """
    Send email report. Returns True if sent successfully.
    Password can come from config or SEO_AGENT_SMTP_PASSWORD env var.
    """
    import os

    email_cfg = config.get("email", {})
    if not email_cfg.get("enabled") or not email_cfg.get("smtp_user"):
        return False

    password = email_cfg.get("smtp_password") or os.environ.get("SEO_AGENT_SMTP_PASSWORD")
    if not password:
        print("Email skipped: Set smtp_password in config or SEO_AGENT_SMTP_PASSWORD env var")
        return False

    subject = f"[SEO Agent] ChangeImageTo.com – {len(implemented)} changes applied"
    if not implemented:
        subject = "[SEO Agent] ChangeImageTo.com – Report (no changes)"

    body = _build_report_html(tactics, implemented, insights or {})
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = email_cfg.get("from_email") or email_cfg.get("smtp_user")
    msg["To"] = email_cfg.get("to_email", "")

    msg.attach(MIMEText(body, "html"))

    if dry_run:
        print("[DRY RUN] Would send email to", email_cfg.get("to_email"))
        return True

    try:
        with smtplib.SMTP(email_cfg["smtp_host"], email_cfg.get("smtp_port", 587)) as server:
            server.starttls()
            server.login(email_cfg["smtp_user"], password)
            server.sendmail(msg["From"], email_cfg["to_email"], msg.as_string())
        return True
    except Exception as e:
        print(f"Email send failed: {e}")
        return False


def _build_report_html(tactics: list, implemented: list, insights: dict) -> str:
    html = """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>
body{font-family:system-ui,sans-serif;max-width:700px;margin:20px auto;padding:20px;color:#333}
h1{color:#1a73e8;font-size:20px}
h2{font-size:16px;margin-top:24px}
h3{font-size:14px;margin-top:16px;color:#5f6368}
table{width:100%;border-collapse:collapse;margin:12px 0}
th,td{border:1px solid #ddd;padding:8px;text-align:left}
th{background:#f5f5f5}
.success{color:#0f9d58}
.skip{color:#5f6368}
ul{margin:8px 0;padding-left:20px}
.insight{background:#f8f9fa;padding:12px;margin:8px 0;border-radius:8px}
</style></head>
<body>
<h1>SEO Agent Report – ChangeImageTo.com</h1>
<p>Automated analysis of GA4 + Google Search Console with tactical implementations.</p>
"""

    if implemented:
        html += "<h2>Changes applied</h2><table><tr><th>Page</th><th>Action</th><th>Result</th></tr>"
        for imp in implemented:
            status = "success" if imp.get("success") else "skip"
            fc = imp.get("file_changed", "")
            fc_str = ", ".join(fc) if isinstance(fc, list) else (fc or "")
            html += f"<tr><td>{fc_str}</td><td>{imp.get('tactic','')}</td><td class='{status}'>{imp.get('message','')}</td></tr>"
        html += "</table>"

    # Insights sections
    if insights.get("conflicts"):
        html += "<h2>Keyword conflicts (cannibalization)</h2><p>Pages competing for the same query. Consider merge or 301 redirect.</p>"
        for c in insights["conflicts"][:8]:
            html += f"<div class='insight'><strong>{c['query']}</strong> – Winner: {c['winner']} ({c['winner_impressions']} impr). Losers: "
            html += ", ".join(f"{l['path']} ({l['impressions']} impr)" for l in c["losers"][:3])
            html += f". <em>{c['recommendation']}</em></div>"

    if insights.get("low_hanging"):
        html += "<h2>Low-hanging fruit (position 31+)</h2><p>Keywords on page 4+ ripe for top 1–3 with small optimizations.</p><ul>"
        for lh in insights["low_hanging"][:8]:
            html += f"<li><strong>{lh['keyword']}</strong> – pos {lh['position']}, {lh['impressions']} impr → {lh['target_page']}</li>"
        html += "</ul>"

    if insights.get("onpage_gaps"):
        html += "<h2>On-page gaps</h2><p>Top queries missing from title, meta, or H1.</p><ul>"
        for g in insights["onpage_gaps"][:8]:
            html += f"<li><strong>{g['query']}</strong> ({g['impressions']} impr) – missing from {', '.join(g['missing_in'])} on {g['path']}</li>"
        html += "</ul>"

    if insights.get("clusters"):
        html += "<h2>Keyword clusters</h2><p>Topic performance. Thin clusters = content opportunities.</p><ul>"
        for cl in insights["clusters"][:10]:
            top_kws = ", ".join(k["query"] for k in cl["keywords"][:3])
            html += f"<li><strong>{cl['cluster']}</strong> – {cl['total_impressions']} impr, {cl['total_clicks']} clicks ({cl['keyword_count']} kws). Top: {top_kws}</li>"
        html += "</ul>"

    if insights.get("competitor_gaps"):
        html += "<h2>Competitor analysis (pages 1–3)</h2><p>Our ranking vs top competitors. Where we lag and how to fix.</p>"
        for cg in insights["competitor_gaps"][:8]:
            our = cg.get("our_position") or "—"
            status = "✓ Page 1" if cg.get("on_page1") else ("Page 2–3" if cg.get("on_page2_3") else "Not in top 30")
            html += f"<div class='insight'><strong>{cg['query']}</strong> ({cg.get('impressions',0)} impr) – We're #{our} ({status})<br>"
            for c in cg.get("top3_competitors", [])[:3]:
                html += f"<small>#{c['position']} {c['domain']}: {c.get('title','')[:60]}…</small><br>"
            html += f"<em>{cg.get('recommendation','')}</em></div>"

    if insights.get("blog_gaps"):
        html += "<h2>Blog creation queue</h2><p>Keywords to create posts for (competitor gaps + GSC gaps, duplicates excluded).</p><ul>"
        for bg in insights["blog_gaps"][:8]:
            src = "competitor" if bg.get("source") == "competitor" else "gsc"
            html += f"<li><strong>{bg['query']}</strong> – {bg['impressions']} impr, {bg['clicks']} clicks <em>({src})</em></li>"
        html += "</ul>"

    if insights.get("blog_created"):
        html += "<h2>Blog posts created</h2><table><tr><th>Topic</th><th>Title</th><th>File</th></tr>"
        for bc in insights["blog_created"]:
            html += f"<tr><td>{bc.get('query','')}</td><td>{bc.get('title','')}</td><td>{bc.get('path','')}</td></tr>"
        html += "</table>"

    html += "<h2>All tactics considered</h2><ul>"
    for t in tactics:
        html += f"<li><strong>{t.get('priority','')}</strong> – {t.get('description','')}</li>"
    html += "</ul>"

    html += "<p><small>Generated by SEO Agent. Run manually or via cron.</small></p></body></html>"
    return html
