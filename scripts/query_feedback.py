#!/usr/bin/env python3
"""
Query user feedback from the database
"""

import sqlite3
import sys
import os

# Add parent directory to path to import from backend if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def query_feedback(db_path='blog_management.db', limit=50):
    """Query feedback from database"""
    if not os.path.exists(db_path):
        print(f"❌ Database file '{db_path}' not found.")
        print("   The database is created when the backend service starts.")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_feedback'")
        if not cursor.fetchone():
            print("⚠️  user_feedback table does not exist yet.")
            print("   It will be created when the backend service starts.")
            conn.close()
            return
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM user_feedback")
        total_count = cursor.fetchone()[0]
        
        print(f"📊 Total feedback entries: {total_count}\n")
        
        if total_count == 0:
            print("No feedback entries found yet.")
            conn.close()
            return
        
        # Get average rating
        cursor.execute("SELECT AVG(rating) FROM user_feedback")
        avg_rating = cursor.fetchone()[0]
        print(f"⭐ Average rating: {avg_rating:.2f}/5\n")
        
        # Get feedback entries
        cursor.execute('''
            SELECT id, rating, comment, page, operation, created_at
            FROM user_feedback
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        print(f"📝 Recent feedback entries (showing {len(rows)} of {total_count}):\n")
        print("=" * 80)
        
        for row in rows:
            print(f"\nID: {row[0]}")
            print(f"Rating: {'⭐' * row[1]} ({row[1]}/5)")
            print(f"Page: {row[3] or '(unknown)'}")
            print(f"Operation: {row[4] or '(unknown)'}")
            print(f"Date: {row[5]}")
            if row[2]:
                comment = row[2]
                if len(comment) > 200:
                    comment = comment[:200] + "..."
                print(f"Comment: {comment}")
            else:
                print("Comment: (no comment)")
            print("-" * 80)
        
        # Get rating distribution
        print("\n📈 Rating Distribution:")
        cursor.execute('''
            SELECT rating, COUNT(*) as count
            FROM user_feedback
            GROUP BY rating
            ORDER BY rating DESC
        ''')
        for row in cursor.fetchall():
            bar = "█" * row[1]
            print(f"  {row[0]} stars: {bar} ({row[1]})")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error querying database: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Query user feedback from database')
    parser.add_argument('--db', default='blog_management.db', help='Database file path')
    parser.add_argument('--limit', type=int, default=50, help='Maximum number of entries to show')
    args = parser.parse_args()
    
    query_feedback(args.db, args.limit)
