import os
import sys
import asyncio
import logging
from flask import Flask, request, jsonify
import json
from datetime import datetime
from google.cloud import storage
import glob

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from scraping.utr_scraper import UTRScraper
except ImportError:
    # Fallback for different path structures
    sys.path.insert(0, '/app/src')
    from scraping.utr_scraper import UTRScraper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get credentials from environment variables
email = os.environ.get('UTR_EMAIL')
password = os.environ.get('UTR_PASSWORD')

@app.route('/')
def health_check():
    """Health check endpoint"""
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', 'tennis-utr-scraper')
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'UTR Tennis Data Scraper',
        'cloud_storage_bucket': f"{project_id}-utr-data",
        'max_processes': os.environ.get('MAX_PROCESSES', '16'),
        'next_scheduled_run': '02:00 EST daily'
    })

@app.route('/results')
def list_results():
    """List available result files in Cloud Storage"""
    try:
        project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', 'tennis-utr-scraper')
        client = storage.Client(project=project_id)
        bucket_name = f"{project_id}-utr-data"
        bucket = client.bucket(bucket_name)
        
        blobs = list(bucket.list_blobs(prefix="utr_data/"))
        files = [{'name': blob.name, 'size': blob.size, 'updated': blob.updated.isoformat()} for blob in blobs]
        
        return jsonify({
            'bucket': bucket_name,
            'total_files': len(files),
            'files': files[-20:],  # Show last 20 files
            'download_url': f"https://console.cloud.google.com/storage/browser/{bucket_name}/utr_data"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-browser', methods=['GET'])
def test_browser():
    """Quick test to verify Playwright browser works"""
    try:
        import asyncio
        result = asyncio.run(test_playwright())
        return jsonify({
            'status': 'success',
            'message': 'Playwright browser test passed',
            'result': result,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'message': f'Browser test failed: {str(e)}',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

async def test_playwright():
    """Simple Playwright test"""
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto('https://www.google.com')
        title = await page.title()
        await browser.close()
        return f"Successfully opened Google, title: {title}"

@app.route('/scrape', methods=['GET', 'POST'])
def trigger_scrape():
    """Trigger UTR data scraping"""
    try:
        # Run async scraping function
        result = asyncio.run(run_scraper())
        return jsonify({
            'status': 'success',
            'message': 'UTR data collection completed',
            'timestamp': datetime.utcnow().isoformat(),
            'result': result
        })
    except Exception as e:
        logger.error(f"Scraping failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/cron/daily-scrape', methods=['GET'])
def cron_scrape():
    """Endpoint for scheduled scraping via Cloud Scheduler"""
    # Verify request is from Cloud Scheduler
    if request.headers.get('X-Appengine-Cron') != 'true':
        return 'Unauthorized', 401
    
    try:
        result = asyncio.run(run_scraper())
        logger.info(f"Scheduled scrape completed: {result}")
        return 'OK', 200
    except Exception as e:
        logger.error(f"Scheduled scrape failed: {str(e)}")
        return 'Error', 500

async def run_scraper():
    """Run the UTR scraper with cloud-optimized settings"""
    if not email or not password:
        raise ValueError("UTR credentials not configured in environment variables")
    
    # Get max processes from environment (default to 16 for cloud)
    max_processes = int(os.environ.get('MAX_PROCESSES', '16'))
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', 'tennis-utr-scraper')
    
    logger.info(f"🚀 Starting cloud UTR scraper with {max_processes} processes")
    
    # Configure scraper with cloud-optimized settings
    async with UTRScraper(
        email=email, 
        password=password, 
        headless=True
    ) as scraper:
        
        if not await scraper.login():
            raise Exception("Failed to log in to UTR website")
        
        logger.info("🎾 Starting comprehensive UTR data collection...")
        logger.info("📊 Target: Top 100 ATP + 1st/2nd degree opponents")
        
        # Use the sophisticated finish_processing pipeline  
        from src.finish_processing import complete_processing
        logger.info(f"🔧 Using {max_processes} processes for data collection")
        if os.environ.get('TEST_SINGLE_PLAYER', '').lower() == 'true':
            logger.info("🧪 TEST MODE: Processing single player only")
        await complete_processing()
        
        logger.info("✅ UTR data collection completed - uploading to Cloud Storage")
        
        # Upload results to Google Cloud Storage
        uploaded_files = await upload_results_to_cloud_storage(project_id)
        
        logger.info(f"☁️ Successfully uploaded {len(uploaded_files)} files to Cloud Storage")
        return f"Data collection completed. {len(uploaded_files)} files uploaded to cloud."

async def upload_results_to_cloud_storage(project_id):
    """Upload scraped data files to Google Cloud Storage"""
    try:
        # Initialize Cloud Storage client
        client = storage.Client(project=project_id)
        bucket_name = f"{project_id}-utr-data"
        
        # Create bucket if it doesn't exist
        try:
            bucket = client.bucket(bucket_name)
            if not bucket.exists():
                bucket = client.create_bucket(bucket_name, location="US")
        except Exception:
            bucket = client.bucket(bucket_name)
        
        # Find all CSV files in the data directory
        data_files = glob.glob('/tmp/data/**/*.csv', recursive=True)
        uploaded_files = []
        
        for file_path in data_files:
            if os.path.getsize(file_path) > 0:  # Only upload non-empty files
                # Create blob name with timestamp
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                filename = os.path.basename(file_path)
                blob_name = f"utr_data/{timestamp}/{filename}"
                
                # Upload file
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(file_path)
                uploaded_files.append(blob_name)
                logger.info(f"📤 Uploaded: {blob_name}")
        
        return uploaded_files
        
    except Exception as e:
        logger.error(f"❌ Cloud Storage upload failed: {str(e)}")
        return []

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)