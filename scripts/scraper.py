from requests_html import AsyncHTMLSession
from bs4 import BeautifulSoup
import nest_asyncio
import asyncio
import csv
import os
import shutil
import sys
from pathlib import Path

# Import config
sys.path.append(str(Path(__file__).parent.parent))
from config import PRODUCT_URL, SCRAPING, SELECTORS

# ✅ Auto-detect Chrome or Edge
chrome_path = (
    shutil.which("chrome")
    or shutil.which("chrome.exe")
    or shutil.which("msedge")
    or shutil.which("msedge.exe")
    or r"C:\Program Files\Google\Chrome\Application\chrome.exe"
)
os.environ["PYPPETEER_BROWSER_PATH"] = chrome_path

nest_asyncio.apply()

# Browser headers to avoid blocking
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Cache-Control': 'max-age=0'
}

async def fetch_page(asession, base_url, page, max_retries=3):
    url = base_url + str(page)
    print(f"PROGRESS|Fetching page {page}...", flush=True)
    page_data = []
    
    for attempt in range(max_retries):
        try:
            r = await asession.get(url, headers=HEADERS, timeout=30)

            # Try rendering using your installed Chrome - wait longer for reviews to load
            try:
                await r.html.arender(timeout=40, sleep=5, wait=3, scrolldown=2)
            except Exception as render_err:
                # Silently continue with static HTML - no need to show skip message
                pass

            soup = BeautifulSoup(r.html.html, "html.parser")
            
            # Use selectors from config
            reviews = [div.get_text(strip=True) for div in soup.find_all("div", class_="ZmyHeo")]
            ratings = [div.get_text(strip=True) for div in soup.find_all("div", class_="XQDdHH")]
            titles = [p.get_text(strip=True) for p in soup.find_all("p", class_="z9E0IG")]
            names = [p.get_text(strip=True) for p in soup.find_all("p", class_="_2NsDsF AwS1CA")]

            for i in range(len(reviews)):
                page_data.append([
                    names[i] if i < len(names) else "N/A",
                    titles[i] if i < len(titles) else "N/A",
                    ratings[i] if i < len(ratings) else "N/A",
                    reviews[i]
                ])
            
            # Always show the page result, even if 0 reviews found
            print(f"PROGRESS|Page {page}: {len(reviews)} reviews found.", flush=True)
            break  # Success, exit retry loop
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                print(f"PROGRESS|Retry {attempt + 1} for page {page} after {wait_time}s...", flush=True)
                await asyncio.sleep(wait_time)
            else:
                print(f"ERROR|Error fetching page {page} after {max_retries} attempts: {str(e)[:100]}", flush=True)
    
    return page_data

async def scrape_flipkart_reviews(total_pages=None, batch_size=None, delay=None, target_reviews=None):
    asession = AsyncHTMLSession()
    
    # Build URL from config
    base_url = (
        f"{PRODUCT_URL['base']}"
        f"{PRODUCT_URL['product_path']}"
        f"{PRODUCT_URL['product_id']}&page="
    )
    
    # Use parameters or config settings
    total_pages = total_pages or SCRAPING["total_pages"]
    batch_size = batch_size or SCRAPING["batch_size"]
    delay_between_batches = delay or SCRAPING["delay_between_batches"]
    target_reviews = target_reviews or 200
    all_data = []
    
    print(f"PROGRESS|Starting scraper: Target {target_reviews} reviews, Max {total_pages} pages in {total_pages//batch_size} batches", flush=True)
    
    for batch_start in range(1, total_pages + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, total_pages)
        print(f"PROGRESS|Starting batch {batch_start}-{batch_end}...", flush=True)
        
        # Fetch pages sequentially to avoid browser overload
        batch_results = []
        for page in range(batch_start, batch_end + 1):
            page_data = await fetch_page(asession, base_url, page)
            batch_results.append(page_data)
            # Small delay between pages
            if page < batch_end:
                await asyncio.sleep(2)
        
        for page_data in batch_results:
            all_data.extend(page_data)
        
        # Calculate progress percentage based on reviews collected vs target
        current_reviews = len(all_data)
        progress_percent = min(int((current_reviews / target_reviews) * 100), 100)
        
        print(f"PROGRESS|{progress_percent}%|{current_reviews} reviews|Completed batch {batch_start}-{batch_end}", flush=True)
        
        # Check if we've reached or exceeded target
        if current_reviews >= target_reviews:
            print(f"PROGRESS|Target reached! Collected {current_reviews} reviews (target: {target_reviews})", flush=True)
            break
        
        # Continue to next batch if target not reached and more pages available
        if batch_end < total_pages:
            print(f"PROGRESS|Waiting {delay_between_batches} seconds before next batch...", flush=True)
            await asyncio.sleep(delay_between_batches)
    
    # Save to data folder with correct filename for phase1 and phase2
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    filename = data_dir / "flipkart_boat_raw.csv"
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Title", "Rating", "Review"])
        writer.writerows(all_data)
    
    print(f"SUCCESS|100%|{len(all_data)} reviews|Scraping complete! Saved to {filename.name}", flush=True)
    
    return len(all_data)

if __name__ == "__main__":
    # Support command line arguments for total_pages, batch_size, delay, target_reviews
    # Usage: python scraper.py [total_pages] [batch_size] [delay] [target_reviews]
    import sys
    
    total_pages = int(sys.argv[1]) if len(sys.argv) > 1 else None
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
    delay = int(sys.argv[3]) if len(sys.argv) > 3 else None
    target_reviews = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    asyncio.run(scrape_flipkart_reviews(total_pages, batch_size, delay, target_reviews))
