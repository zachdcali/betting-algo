#!/bin/bash

# UTR Tennis Scraper - Google Cloud Deployment Script
# This script automates the deployment process to Google App Engine

set -e  # Exit on any error

echo "🎾 UTR Tennis Scraper - Cloud Deployment"
echo "========================================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud CLI not found. Please install it first:"
    echo "   curl https://sdk.cloud.google.com | bash"
    exit 1
fi

# Set project variables
PROJECT_ID="tennis-utr-scraper"
REGION="us-central1"

echo "📋 Project Configuration:"
echo "   Project ID: $PROJECT_ID"
echo "   Region: $REGION"
echo ""

# Create project if it doesn't exist
echo "🔧 Setting up Google Cloud project..."
if ! gcloud projects describe $PROJECT_ID &>/dev/null; then
    echo "   Creating new project: $PROJECT_ID"
    gcloud projects create $PROJECT_ID --name="Tennis UTR Scraper"
else
    echo "   Project $PROJECT_ID already exists"
fi

# Set active project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔌 Enabling required APIs..."
gcloud services enable appengine.googleapis.com --quiet
gcloud services enable cloudscheduler.googleapis.com --quiet
gcloud services enable logging.googleapis.com --quiet
gcloud services enable storage.googleapis.com --quiet

# Initialize App Engine if not already done
echo "🚀 Initializing App Engine..."
if ! gcloud app describe &>/dev/null; then
    gcloud app create --region=$REGION
else
    echo "   App Engine already initialized"
fi

# Prompt for UTR credentials
echo ""
echo "🔐 UTR Credentials Setup"
echo "Please enter your UTR credentials (they will be stored as environment variables):"
read -p "UTR Email: " UTR_EMAIL
read -s -p "UTR Password: " UTR_PASSWORD
echo ""

# Update app.yaml with credentials
echo "📝 Updating configuration with credentials..."
sed -i.bak "s/your-email@example.com/$UTR_EMAIL/g" app.yaml
sed -i.bak "s/your-password/$UTR_PASSWORD/g" app.yaml

# Deploy application
echo ""
echo "☁️ Deploying application to Google Cloud..."
gcloud app deploy app.yaml --quiet

# Deploy cron jobs
echo "⏰ Setting up automated scheduling..."
gcloud app deploy cron.yaml --quiet

# Get application URL
APP_URL=$(gcloud app describe --format="value(defaultHostname)")

echo ""
echo "✅ Deployment completed successfully!"
echo "========================================"
echo "🌐 Application URL: https://$APP_URL"
echo "🔍 Health Check: https://$APP_URL/"
echo "🎯 Manual Trigger: https://$APP_URL/scrape"
echo "📁 View Results: https://$APP_URL/results"
echo ""
echo "☁️ Data Storage:"
echo "   Cloud Storage Bucket: $PROJECT_ID-utr-data"
echo "   Download Files: https://console.cloud.google.com/storage/browser/$PROJECT_ID-utr-data/utr_data"
echo ""
echo "📊 Monitoring Commands:"
echo "   View logs: gcloud app logs tail -s default"
echo "   Open browser: gcloud app browse"
echo "   Access storage: gcloud storage ls gs://$PROJECT_ID-utr-data/"
echo ""
echo "⚡ Enhanced Cloud Features:"
echo "   • 16 concurrent processes (vs 8 locally)"
echo "   • Automated daily collection at 2:00 AM EST"  
echo "   • All results automatically saved to Cloud Storage"
echo "   • Email alerts for any errors"
echo "   • Professional progress logging with emojis"

# Clean up backup file
rm -f app.yaml.bak