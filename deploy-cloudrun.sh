#!/bin/bash

# Deploy to Google Cloud Run
# Prerequisites: gcloud CLI installed and authenticated

set -e

PROJECT_ID="vipul510"  # Your GCP project ID
SERVICE_NAME="bgremover-backend"
REGION="us-central1"

echo "Building and deploying to Cloud Run..."

# Build and deploy
gcloud run deploy $SERVICE_NAME \
  --source . \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --concurrency 2 \
  --min-instances 0 \
  --max-instances 10 \
  --timeout 300 \
  --port 8080 \
  --set-env-vars DEFAULT_MODEL=u2netp,MAX_IMAGE_SIDE=1600,MAX_CONCURRENCY=2,BLOG_BUCKET=${BLOG_BUCKET:-changeimageto-blog},CRON_TOKEN=${CRON_TOKEN:-changeme},PIXABAY_API_KEY=53808557-ae026e819afe0e03779db078c

echo "Deployment complete!"
echo "Service URL: https://$SERVICE_NAME-$REGION.a.run.app"
