#!/bin/bash
# Daily Analytics Runner for Background Removal Tool
# Run this script daily to get usage analytics

echo "🔍 Running daily analytics for background removal tool..."
echo "Date: $(date)"
echo ""

# Check if analytics.py exists
if [ ! -f "analytics.py" ]; then
    echo "❌ analytics.py not found!"
    exit 1
fi

DAYS=${1:-1}  # Default to 1 day if not specified

# Check if we should fetch Cloud Run logs
# If app.log doesn't exist or is older than 1 day, try fetching from Cloud Run
if [ ! -f "app.log" ] || [ $(find app.log -mtime +1 2>/dev/null | wc -l) -gt 0 ]; then
    echo "📥 No recent local logs found. Attempting to fetch from Cloud Run..."
    if [ -f "scripts/fetch_cloudrun_logs.sh" ]; then
        ./scripts/fetch_cloudrun_logs.sh $DAYS
        if [ $? -eq 0 ]; then
            echo "✅ Cloud Run logs fetched successfully"
        else
            echo "⚠️  Could not fetch Cloud Run logs. Using local logs if available."
        fi
    else
        echo "⚠️  Cloud Run log fetcher not found. Using local logs if available."
    fi
fi

# Run analytics for specified days
echo "📊 Analyzing last $DAYS day(s)..."
python3 analytics.py --days $DAYS

echo ""
echo "✅ Analytics complete!"
echo ""
echo "💡 Tips:"
echo "  - Run this script daily to track usage patterns"
echo "  - Use './run_analytics.sh 7' for weekly analysis"
echo "  - Use './scripts/fetch_cloudrun_logs.sh 30' to fetch Cloud Run logs manually"
echo "  - Check the generated summary files for historical data"
