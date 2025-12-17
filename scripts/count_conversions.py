#!/usr/bin/env python3
"""
Count image conversions from log files for the last N days.
"""

import json
import re
from datetime import datetime, timedelta
from collections import Counter
import os
import sys

# Conversion action types that indicate a successful conversion
CONVERSION_ACTIONS = [
    "processing_completed",
    "color_change_processing_completed",
    "convert_completed",
    "upscale_completed",
    "blur_background_completed",
    "enhance_image_completed",
    "text_removal_success",
    "bulk_resize_completed",
    "bulk_convert_completed",
    "painted_areas_removal_success"
]

def parse_log_file(log_file='app.log', days=30):
    """Parse the log file and count conversions in the last N days"""
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found!")
        return 0, []
    
    cutoff_date = datetime.now() - timedelta(days=days)
    conversions = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'USER_ACTION:' in line:
                try:
                    # Extract JSON from log line
                    json_start = line.find('USER_ACTION: ') + len('USER_ACTION: ')
                    json_str = line[json_start:].strip()
                    action_data = json.loads(json_str)
                    
                    action = action_data.get('action', '')
                    
                    # Check if this is a conversion action
                    if action in CONVERSION_ACTIONS:
                        # Parse timestamp
                        timestamp_str = action_data.get('timestamp', '')
                        try:
                            # Handle different timestamp formats
                            if 'T' in timestamp_str:
                                if timestamp_str.endswith('Z'):
                                    action_date = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                else:
                                    action_date = datetime.fromisoformat(timestamp_str)
                            else:
                                # Try parsing from log line timestamp
                                log_timestamp = line.split(' - ')[0]
                                action_date = datetime.strptime(log_timestamp, '%Y-%m-%d %H:%M:%S,%f')
                            
                            if action_date >= cutoff_date:
                                conversions.append({
                                    'action': action,
                                    'timestamp': action_date,
                                    'details': action_data.get('details', {})
                                })
                        except (ValueError, KeyError) as e:
                            # If timestamp parsing fails, try to use log line timestamp
                            try:
                                log_timestamp = line.split(' - ')[0]
                                action_date = datetime.strptime(log_timestamp, '%Y-%m-%d %H:%M:%S,%f')
                                if action_date >= cutoff_date:
                                    conversions.append({
                                        'action': action,
                                        'timestamp': action_date,
                                        'details': action_data.get('details', {})
                                    })
                            except:
                                pass
                except (json.JSONDecodeError, ValueError) as e:
                    continue
    
    return len(conversions), conversions

def main():
    days = 30
    log_file = 'app.log'
    
    if len(sys.argv) > 1:
        days = int(sys.argv[1])
    if len(sys.argv) > 2:
        log_file = sys.argv[2]
    
    count, conversions = parse_log_file(log_file, days)
    
    print("=" * 60)
    print(f"IMAGE CONVERSIONS - LAST {days} DAYS")
    print("=" * 60)
    print(f"Total conversions: {count}")
    print()
    
    if conversions:
        # Breakdown by action type
        action_counts = Counter(c['action'] for c in conversions)
        print("Breakdown by conversion type:")
        for action_type, action_count in action_counts.most_common():
            print(f"  {action_type}: {action_count}")
        print()
        
        # Daily breakdown
        daily_counts = Counter(c['timestamp'].date() for c in conversions)
        print("Daily breakdown (last 10 days with activity):")
        for date, day_count in sorted(daily_counts.items(), reverse=True)[:10]:
            print(f"  {date}: {day_count}")
        print()
        
        # Most recent conversions
        print("Most recent 5 conversions:")
        for conv in sorted(conversions, key=lambda x: x['timestamp'], reverse=True)[:5]:
            print(f"  {conv['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {conv['action']}")
    else:
        print(f"No conversions found in the last {days} days.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

