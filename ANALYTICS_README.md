# Background Removal Tool - Analytics Guide

## Overview
This tool now includes comprehensive logging to track user behavior and usage patterns. You can run analytics daily to understand how users interact with your site.

## What Gets Logged

### Backend Logging (Server-side)
- **Request details**: IP address, user agent, filename, file size
- **Processing requests**: Category selected, target color, action type
- **File processing**: Image dimensions, file sizes
- **Success/failure**: Processing completion status and error details
- **Performance**: Processing times and output sizes

### Frontend Logging (Client-side)
- **Page visits**: Which pages users visit and how they arrived
- **File uploads**: File details and user interactions
- **Processing**: Start/completion of background removal/color changes
- **Downloads**: When users download processed images
- **Errors**: Client-side errors and issues
- **User actions**: Button clicks, resets, color selections

## Running Analytics

### Quick Daily Analysis
```bash
./run_analytics.sh
```

### Custom Analysis
```bash
# Analyze last 7 days
python3 analytics.py --days 7

# Analyze specific log file
python3 analytics.py --log-file /path/to/app.log --days 1
```

## Analytics Output

The script provides detailed reports including:

### 📊 Action Breakdown
- Total number of each action type
- Page visits by type (remove_background, color_specific, color_palette)
- File uploads with size analysis
- Processing requests and success rates

### ⚙️ Processing Analysis
- Success/failure rates
- Most popular categories (product vs portrait)
- Most used colors for background changes
- Processing performance metrics

### 🌐 User Behavior
- Browser usage patterns
- Hourly usage distribution
- File type preferences
- Error analysis

### 📁 File Analysis
- Average file sizes
- File type distribution
- Upload patterns

## Log Files

- **`app.log`**: Main log file with all user actions
- **`analytics_summary_YYYYMMDD.txt`**: Daily summary files

## Setting Up Daily Analytics

### Option 1: Manual Daily Check
Run the analytics script daily:
```bash
./run_analytics.sh
```

### Option 2: Automated Daily Reports
Add to your crontab for daily automated reports:
```bash
# Add this line to crontab (crontab -e)
0 9 * * * cd /path/to/your/project && ./run_analytics.sh >> daily_analytics.log 2>&1
```

### Option 3: Weekly Analysis
For weekly trends:
```bash
python3 analytics.py --days 7
```

## Key Metrics to Monitor

### Daily Metrics
- **Total page visits**: Track overall traffic
- **File uploads**: Measure user engagement
- **Processing success rate**: Monitor tool reliability
- **Most popular colors**: Understand user preferences
- **Error rates**: Identify and fix issues

### Weekly Trends
- **Usage patterns**: Peak hours and days
- **Browser compatibility**: Ensure cross-browser support
- **File size trends**: Optimize for user needs
- **Feature usage**: Which pages/features are most popular

## Sample Analytics Output

```
============================================================
BACKGROUND REMOVAL TOOL - DAILY ANALYTICS REPORT
============================================================
Report generated: 2025-09-15 14:30:00
Total actions analyzed: 150

📊 ACTION BREAKDOWN:
  page_visit: 45
  file_uploaded: 32
  processing_started: 30
  processing_completed: 28
  download_link_created: 28
  reset_action: 8
  processing_error: 2

🌐 PAGE VISITS:
  remove_background: 20
  color_specific: 15
  color_palette: 10

📁 FILE UPLOADS:
  Total uploads: 32
  Average file size: 2.45 MB
  Largest file: 8.2 MB
  Smallest file: 0.1 MB
  File types:
    image/jpeg: 18
    image/png: 12
    image/webp: 2

⚙️ PROCESSING REQUESTS:
  Total processing requests: 30
  Successful completions: 28
  Errors: 2
  Success rate: 93.3%
  Action types:
    remove_background: 18
    change_background: 12
  Categories:
    product: 22
    portrait: 8
  Most popular colors:
    #FFFFFF: 5
    #000000: 3
    #FF0000: 2
```

## Troubleshooting

### No Log Data
- Ensure the backend is running and processing requests
- Check that `app.log` exists and has recent entries
- Verify frontend is sending analytics data

### Missing Analytics
- Check browser console for JavaScript errors
- Ensure `/api/analytics` endpoint is accessible
- Verify CORS settings allow frontend requests

### Performance Impact
- Logging is designed to be lightweight
- Frontend analytics fail silently if backend is unavailable
- Log files are automatically rotated (you may want to implement log rotation)

## Privacy Considerations

The logging system collects:
- IP addresses (for basic analytics)
- User agents (for browser analysis)
- File metadata (size, type, dimensions)
- Usage patterns (pages visited, actions taken)

**No personal data or image content is logged.**

Consider implementing:
- IP anonymization
- Log retention policies
- GDPR compliance measures if needed
