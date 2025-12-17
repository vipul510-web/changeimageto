#!/bin/bash
# Fetch recent logs from Cloud Run and save to app.log format

PROJECT_ID="vipul510"
SERVICE_NAME="bgremover-backend"
DAYS=${1:-1}  # Default to 1 day if not specified

echo "Fetching logs from Cloud Run service: $SERVICE_NAME"
echo "Project: $PROJECT_ID"
echo "Days: $DAYS"

# Calculate the timestamp for N days ago
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    SINCE=$(date -u -v-${DAYS}d '+%Y-%m-%dT%H:%M:%SZ')
else
    # Linux
    SINCE=$(date -u -d "${DAYS} days ago" '+%Y-%m-%dT%H:%M:%SZ')
fi

echo "Fetching logs since: $SINCE"

# Fetch logs and format them similar to local app.log format
# Filter for application logs (not infrastructure logs) and specifically look for USER_ACTION
# The logs are in stderr/stdout, so we need to filter for USER_ACTION in textPayload
echo "Fetching USER_ACTION logs from Cloud Run..."

# Use JSON format to get the full log entry, then extract textPayload
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME AND timestamp>=\"$SINCE\" AND textPayload=~\"USER_ACTION\"" \
    --project=$PROJECT_ID \
    --format=json \
    --limit=10000 | python3 -c "
import json
import sys

data = json.load(sys.stdin)
for entry in data:
    if 'textPayload' in entry and 'USER_ACTION' in entry['textPayload']:
        # The textPayload already has the correct format: 'YYYY-MM-DD HH:MM:SS,mmm - LEVEL - MESSAGE'
        print(entry['textPayload'])
" > cloudrun_logs_$(date +%Y%m%d).log

# Check if we got any logs
if [ ! -s "cloudrun_logs_$(date +%Y%m%d).log" ] || [ $(grep -c "USER_ACTION" "cloudrun_logs_$(date +%Y%m%d).log" 2>/dev/null || echo 0) -eq 0 ]; then
    echo "No USER_ACTION logs found with JSON format. Trying alternative method..."
    # Fallback: try to get textPayload directly
    gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME AND timestamp>=\"$SINCE\" AND textPayload=~\"USER_ACTION\"" \
        --project=$PROJECT_ID \
        --format="value(textPayload)" \
        --limit=10000 > cloudrun_logs_$(date +%Y%m%d).log
fi

echo "Logs saved to: cloudrun_logs_$(date +%Y%m%d).log"
echo "You can now run: python3 analytics.py --log-file cloudrun_logs_$(date +%Y%m%d).log --days $DAYS"

