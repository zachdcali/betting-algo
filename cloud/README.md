# Legacy UTR Tennis Scraper - Deployment Disabled Pending Review

> **Do not deploy this directory as the betting pipeline.** The active
> production path is the TA-based GitHub Actions workflow documented in
> `docs/production/README.md`. This older Flask/App Engine/Cloud Run service has
> stale vulnerable dependencies, imports a missing scraper module, exposes an
> unauthenticated trigger in its historical configuration, and requires an
> authenticated inventory/retirement decision. See
> `docs/production/SECURITY_AUDIT_2026-07-14.md`.

This directory contains all the necessary files to deploy your UTR tennis data scraper to Google Cloud Platform for reliable, automated data collection.

## 🚀 Quick Start

1. **Install Google Cloud CLI** (if not already installed):
   ```bash
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   gcloud init
   ```

2. **Deploy with one command**:
   ```bash
   cd cloud/
   ./deploy.sh
   ```

The deployment script will:
- Create a Google Cloud project
- Enable required APIs
- Set up App Engine
- Configure automated daily scheduling
- Deploy your scraper

## 📁 File Overview

- **`app.yaml`** - Google App Engine configuration
- **`main.py`** - Flask web service wrapper for the scraper
- **`requirements.txt`** - Python dependencies for cloud deployment
- **`cron.yaml`** - Scheduled job configuration (daily at 2 AM EST)
- **`deploy.sh`** - Automated deployment script
- **`.gcloudignore`** - Files to exclude from deployment

## 🎯 Endpoints

Once deployed, your scraper will have these endpoints:

- **`/`** - Health check endpoint
- **`/scrape`** - Manual trigger for data collection
- **`/cron/daily-scrape`** - Automated daily collection (Cloud Scheduler only)

## 📊 Monitoring

```bash
# View real-time logs
gcloud app logs tail -s default

# Open application in browser
gcloud app browse

# Check scheduled jobs
gcloud scheduler jobs list
```

## 🔧 Configuration

The scraper is configured to collect:
- **Top 100 ATP players**
- **Match history** (2020-2025)
- **UTR ratings** over time
- **Opponent data** for comprehensive analysis

## 💰 Cost Estimation

Google App Engine pricing:
- **Automatic scaling**: 0-1 instances
- **Daily execution**: ~5-10 minutes
- **Expected cost**: $5-15/month
- **Free tier**: 28 instance hours/day

## 🔐 Security

- Credentials stored as environment variables
- HTTPS-only communication
- Access restricted to Cloud Scheduler for automated runs
- No sensitive data logged

## 🛠️ Manual Deployment Steps

If you prefer manual deployment:

```bash
# 1. Create project
gcloud projects create tennis-utr-scraper
gcloud config set project tennis-utr-scraper

# 2. Enable APIs
gcloud services enable appengine.googleapis.com
gcloud services enable cloudscheduler.googleapis.com

# 3. Initialize App Engine
gcloud app create --region=us-central1

# 4. Update credentials in app.yaml
# 5. Deploy
gcloud app deploy app.yaml
gcloud app deploy cron.yaml
```

## 📈 Benefits of Cloud Deployment

1. **24/7 Reliability** - No dependency on your home network
2. **Automated Scheduling** - Daily data collection without manual intervention
3. **Scalability** - Handles traffic spikes and data processing
4. **Monitoring** - Built-in logging and error tracking
5. **Professional Skills** - Adds cloud computing experience to your resume

## 🎾 Data Collection Strategy

The cloud scraper implements the same multi-degree collection strategy as your local version:
- Primary targets: Top 100 ATP players
- Secondary collection: Recent opponents and their networks
- Historical depth: 6 years of match and rating data
- Smart filtering: UTR threshold and dynamic year selection

Your tennis prediction models will now have access to consistently updated, comprehensive UTR data for enhanced accuracy in ATP-level match predictions.
