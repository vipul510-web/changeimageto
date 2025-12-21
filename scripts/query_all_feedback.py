#!/usr/bin/env python3
"""
Query all feedback and review data from the production API
This script queries the live API endpoints to get feedback data
"""

import requests
import json
import sys

API_BASE = "https://bgremover-backend-121350814881.us-central1.run.app"

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def query_feedback():
    """Query image processing feedback"""
    try:
        response = requests.get(f"{API_BASE}/api/feedback?include_stats=true&limit=100")
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return
        
        data = response.json()
        
        print_section("📊 IMAGE PROCESSING FEEDBACK")
        print(f"Total Feedback Entries: {data.get('count', 0)}")
        
        # Show statistics
        if 'stats' in data:
            stats = data['stats']
            print(f"\n📈 IMPRESSION STATISTICS:")
            print(f"  • Impressions Shown: {stats.get('impressions_shown', 0)}")
            print(f"  • Impressions Submitted: {stats.get('impressions_submitted', 0)}")
            print(f"  • Impressions Skipped: {stats.get('impressions_skipped', 0)}")
            print(f"  • Impressions Closed: {stats.get('impressions_closed', 0)}")
            
            conversion_rate = stats.get('conversion_rate', 0)
            if conversion_rate > 0:
                print(f"  • Conversion Rate: {conversion_rate}%")
            else:
                shown = stats.get('impressions_shown', 0)
                if shown > 0:
                    print(f"  • Conversion Rate: 0% (no submissions yet)")
                else:
                    print(f"  • Conversion Rate: N/A (no impressions yet)")
        
        # Show feedback entries
        feedback_list = data.get('feedback', [])
        if feedback_list:
            print(f"\n💬 RECENT FEEDBACK ENTRIES ({len(feedback_list)} shown):")
            
            # Calculate average rating
            ratings = [fb.get('rating', 0) for fb in feedback_list if fb.get('rating')]
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                print(f"\n  Average Rating: {avg_rating:.2f}/5 ({len(ratings)} ratings)")
            
            for i, fb in enumerate(feedback_list, 1):
                print(f"\n  {i}. Entry ID: {fb.get('id')}")
                print(f"     Rating: {'⭐' * fb.get('rating', 0)} ({fb.get('rating', 0)}/5)")
                print(f"     Page: {fb.get('page', 'unknown')}")
                print(f"     Operation: {fb.get('operation', 'unknown')}")
                print(f"     Date: {fb.get('created_at', 'unknown')}")
                
                comment = fb.get('comment', '')
                if comment:
                    if len(comment) > 200:
                        comment = comment[:200] + "..."
                    print(f"     Comment: {comment}")
                else:
                    print(f"     Comment: (no comment)")
        else:
            print("\n  No feedback entries yet.")
        
    except Exception as e:
        print(f"❌ Error querying feedback: {e}")

def query_whats_missing():
    """Query 'what's missing' feedback"""
    try:
        response = requests.get(f"{API_BASE}/api/whats-missing?limit=100")
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return
        
        data = response.json()
        
        print_section("💬 'WHAT'S MISSING' FEEDBACK")
        print(f"Total Entries: {data.get('count', 0)}")
        
        feedback_list = data.get('feedback', [])
        if feedback_list:
            print(f"\n📝 RECENT ENTRIES ({len(feedback_list)} shown):")
            for i, fb in enumerate(feedback_list, 1):
                print(f"\n  {i}. Entry ID: {fb.get('id')}")
                print(f"     Page: {fb.get('page', 'unknown')}")
                print(f"     Date: {fb.get('created_at', 'unknown')}")
                
                feedback_text = fb.get('feedback', '')
                if len(feedback_text) > 300:
                    feedback_text = feedback_text[:300] + "..."
                print(f"     Feedback: {feedback_text}")
        else:
            print("\n  No 'what's missing' feedback entries yet.")
        
    except Exception as e:
        print(f"❌ Error querying 'what's missing' feedback: {e}")

def main():
    print("=" * 80)
    print("FEEDBACK & REVIEW DATA QUERY")
    print("=" * 80)
    print(f"Querying: {API_BASE}")
    print("\nNote: Data is now stored persistently in Cloud Storage!")
    print("      Feedback will survive deployments and can be queried anytime.")
    
    query_feedback()
    query_whats_missing()
    
    print("\n" + "=" * 80)
    print("Query complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
