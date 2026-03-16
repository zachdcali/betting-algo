# Google Cloud Platform Setup for UTR Scraper

## Prerequisites
1. Install Google Cloud CLI: https://cloud.google.com/sdk/docs/install
2. Create Google Cloud account and billing setup
3. Enable required APIs

## Step 1: Install Google Cloud CLI
```bash
# macOS
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

## Step 2: Create and Configure Project
```bash
# Create new project
gcloud projects create tennis-utr-scraper --name="Tennis UTR Scraper"

# Set as active project
gcloud config set project tennis-utr-scraper

# Enable required APIs
gcloud services enable appengine.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
gcloud services enable logging.googleapis.com
```

## Step 3: Initialize App Engine
```bash
gcloud app create --region=us-central1
```

## Step 4: Deploy Application
```bash
# From betting-algo directory
gcloud app deploy cloud/app.yaml
```

## Step 5: Set Up Automated Scheduling
```bash
# Deploy cron jobs
gcloud app deploy cloud/cron.yaml
```

## Monitoring and Logs
```bash
# View logs
gcloud app logs tail -s default

# View app in browser
gcloud app browse
```