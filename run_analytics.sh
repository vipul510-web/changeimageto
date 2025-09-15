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

# Check if log file exists
if [ ! -f "app.log" ]; then
    echo "❌ app.log not found! Make sure the backend is running and has processed some requests."
    exit 1
fi

# Run analytics for last 1 day
echo "📊 Analyzing last 24 hours..."
python3 analytics.py --days 1

echo ""
echo "✅ Analytics complete!"
echo ""
echo "💡 Tips:"
echo "  - Run this script daily to track usage patterns"
echo "  - Use 'python3 analytics.py --days 7' for weekly analysis"
echo "  - Check the generated summary files for historical data"
