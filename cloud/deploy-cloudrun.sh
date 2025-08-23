#!/bin/bash

# UTR Tennis Scraper - Cloud Run Deployment Script
set -e

echo "🎾 UTR Tennis Scraper - Cloud Run Deployment"
echo "============================================="

# Set project variables
PROJECT_ID="tennis-utr-scraper"
SERVICE_NAME="utr-scraper"
REGION="us-central1"

echo "📋 Project Configuration:"
echo "   Project ID: $PROJECT_ID"
echo "   Service: $SERVICE_NAME"
echo "   Region: $REGION"
echo ""

# Set active project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔌 Enabling required APIs..."
gcloud services enable run.googleapis.com --quiet
gcloud services enable cloudbuild.googleapis.com --quiet
gcloud services enable storage.googleapis.com --quiet

# Build and deploy to Cloud Run
echo ""
echo "🏗️  Building and deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --source . \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --max-instances 5 \
    --set-env-vars UTR_EMAIL="zachdodson12@gmail.com" \
    --set-env-vars UTR_PASSWORD="Thailand@123" \
    --set-env-vars GOOGLE_CLOUD_PROJECT="$PROJECT_ID" \
    --set-env-vars MAX_PROCESSES="8" \
    --set-env-vars TEST_SINGLE_PLAYER="true"

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

echo ""
echo "✅ Cloud Run deployment completed successfully!"
echo "=============================================="
echo "🌐 Service URL: $SERVICE_URL"
echo "🔍 Health Check: $SERVICE_URL/"
echo "🎯 Manual Trigger: $SERVICE_URL/scrape"
echo "📁 View Results: $SERVICE_URL/results"
echo ""
echo "📊 Monitoring Commands:"
echo "   View logs: gcloud run logs tail --service=$SERVICE_NAME --region=$REGION"
echo "   Service info: gcloud run services describe $SERVICE_NAME --region=$REGION"
echo ""
echo "⚡ Cloud Run Advantages:"
echo "   • Full GLIBC 2.28+ support for Playwright"
echo "   • 2GB RAM + 2 CPU cores per instance"
echo "   • Pay-per-use (no always-on costs)"
echo "   • 1 hour timeout for long scraping jobs"
echo "   • Auto-scaling up to 5 instances"