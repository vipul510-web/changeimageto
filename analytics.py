#!/usr/bin/env python3
"""
Daily Analytics Script for Background Removal Tool
Run this script daily to analyze user behavior and usage patterns.
"""

import json
import re
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import os

def parse_log_file(log_file='app.log'):
    """Parse the log file and extract user actions"""
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found!")
        return []
    
    actions = []
    with open(log_file, 'r') as f:
        for line in f:
            if 'USER_ACTION:' in line:
                try:
                    # Extract JSON from log line
                    json_start = line.find('USER_ACTION: ') + len('USER_ACTION: ')
                    json_str = line[json_start:].strip()
                    action_data = json.loads(json_str)
                    actions.append(action_data)
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error parsing line: {line.strip()}")
                    continue
    
    return actions

def analyze_actions(actions, days=1):
    """Analyze user actions for the last N days"""
    cutoff_date = datetime.now() - timedelta(days=days)
    
    recent_actions = []
    for action in actions:
        try:
            action_date = datetime.fromisoformat(action['timestamp'].replace('Z', '+00:00'))
            if action_date >= cutoff_date:
                recent_actions.append(action)
        except (ValueError, KeyError):
            continue
    
    return recent_actions

def generate_report(actions):
    """Generate a comprehensive analytics report"""
    if not actions:
        print("No actions found for the specified period.")
        return
    
    print("=" * 60)
    print("BACKGROUND REMOVAL TOOL - DAILY ANALYTICS REPORT")
    print("=" * 60)
    print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total actions analyzed: {len(actions)}")
    print()
    
    # Action type breakdown
    action_types = Counter(action['action'] for action in actions)
    print("📊 ACTION BREAKDOWN:")
    for action_type, count in action_types.most_common():
        print(f"  {action_type}: {count}")
    print()
    
    # Page visits
    page_visits = [a for a in actions if a['action'] == 'page_visit']
    if page_visits:
        print("🌐 PAGE VISITS:")
        page_types = Counter(a['details'].get('page_type', 'unknown') for a in page_visits)
        for page_type, count in page_types.most_common():
            print(f"  {page_type}: {count}")
        print()
    
    # File uploads
    uploads = [a for a in actions if a['action'] == 'file_uploaded']
    if uploads:
        print("📁 FILE UPLOADS:")
        print(f"  Total uploads: {len(uploads)}")
        
        # File size analysis
        sizes = [a['details'].get('file_size', 0) for a in uploads]
        if sizes:
            avg_size = sum(sizes) / len(sizes)
            print(f"  Average file size: {avg_size / (1024*1024):.2f} MB")
            print(f"  Largest file: {max(sizes) / (1024*1024):.2f} MB")
            print(f"  Smallest file: {min(sizes) / (1024*1024):.2f} MB")
        
        # File types
        file_types = Counter(a['details'].get('file_type', 'unknown') for a in uploads)
        print("  File types:")
        for file_type, count in file_types.most_common():
            print(f"    {file_type}: {count}")
        print()
    
    # Processing requests
    processing_started = [a for a in actions if a['action'] == 'processing_started']
    processing_completed = [a for a in actions if a['action'] == 'processing_completed']
    processing_errors = [a for a in actions if a['action'] == 'processing_error']
    
    if processing_started:
        print("⚙️ PROCESSING REQUESTS:")
        print(f"  Total processing requests: {len(processing_started)}")
        print(f"  Successful completions: {len(processing_completed)}")
        print(f"  Errors: {len(processing_errors)}")
        
        if processing_started:
            success_rate = len(processing_completed) / len(processing_started) * 100
            print(f"  Success rate: {success_rate:.1f}%")
        
        # Action types (remove vs change background)
        action_types = Counter(a['details'].get('action_type', 'unknown') for a in processing_started)
        print("  Action types:")
        for action_type, count in action_types.most_common():
            print(f"    {action_type}: {count}")
        
        # Categories
        categories = Counter(a['details'].get('category', 'unknown') for a in processing_started)
        print("  Categories:")
        for category, count in categories.most_common():
            print(f"    {category}: {count}")
        
        # Target colors
        colors = [a['details'].get('target_color') for a in processing_started if a['details'].get('target_color')]
        if colors:
            color_counts = Counter(colors)
            print("  Most popular colors:")
            for color, count in color_counts.most_common(5):
                print(f"    {color}: {count}")
        print()
    
    # Downloads
    downloads = [a for a in actions if a['action'] == 'download_link_created']
    if downloads:
        print("⬇️ DOWNLOADS:")
        print(f"  Total downloads: {len(downloads)}")
        print()
    
    # Errors analysis
    if processing_errors:
        print("❌ ERROR ANALYSIS:")
        error_types = Counter(a['details'].get('error_type', 'unknown') for a in processing_errors)
        for error_type, count in error_types.most_common():
            print(f"  {error_type}: {count}")
        print()
    
    # Hourly distribution
    hours = []
    for action in actions:
        try:
            action_date = datetime.fromisoformat(action['timestamp'].replace('Z', '+00:00'))
            hours.append(action_date.hour)
        except:
            continue
    
    if hours:
        print("🕐 HOURLY DISTRIBUTION:")
        hour_counts = Counter(hours)
        for hour in sorted(hour_counts.keys()):
            count = hour_counts[hour]
            bar = "█" * (count // max(1, max(hour_counts.values()) // 20))
            print(f"  {hour:2d}:00 {bar} {count}")
        print()
    
    # User agents (browser analysis)
    user_agents = [a.get('userAgent', '') for a in actions if a.get('userAgent')]
    if user_agents:
        print("🌐 BROWSER ANALYSIS:")
        browsers = defaultdict(int)
        for ua in user_agents:
            if 'Chrome' in ua:
                browsers['Chrome'] += 1
            elif 'Firefox' in ua:
                browsers['Firefox'] += 1
            elif 'Safari' in ua:
                browsers['Safari'] += 1
            elif 'Edge' in ua:
                browsers['Edge'] += 1
            else:
                browsers['Other'] += 1
        
        for browser, count in sorted(browsers.items(), key=lambda x: x[1], reverse=True):
            print(f"  {browser}: {count}")
        print()

def main():
    """Main function to run the analytics"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze background removal tool usage')
    parser.add_argument('--days', type=int, default=1, help='Number of days to analyze (default: 1)')
    parser.add_argument('--log-file', default='app.log', help='Log file to analyze (default: app.log)')
    
    args = parser.parse_args()
    
    # Parse and analyze
    all_actions = parse_log_file(args.log_file)
    recent_actions = analyze_actions(all_actions, args.days)
    
    if not recent_actions:
        print(f"No actions found in the last {args.days} day(s).")
        return
    
    generate_report(recent_actions)
    
    # Save summary to file
    summary_file = f"analytics_summary_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Analytics Summary - {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"Total actions: {len(recent_actions)}\n")
        
        action_types = Counter(action['action'] for action in recent_actions)
        f.write("Action breakdown:\n")
        for action_type, count in action_types.most_common():
            f.write(f"  {action_type}: {count}\n")
    
    print(f"📄 Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
