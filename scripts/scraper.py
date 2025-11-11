from requests_html import AsyncHTMLSession
from bs4 import BeautifulSoup
import nest_asyncio
import asyncio
import csv
import os
import shutil
import sys
from pathlib import Path

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

async def fetch_page(asession, base_url, page):
    url = base_url + str(page)
    print(f"PROGRESS|Fetching page {page}...", flush=True)
    page_data = []
    try:
        r = await asession.get(url)

        # Try rendering using your installed Chrome
        try:
            await r.html.arender(timeout=30, sleep=2)
        except Exception as render_err:
            # Silently continue with static HTML - no need to show skip message
            pass

        soup = BeautifulSoup(r.html.html, "html.parser")
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
    except Exception as e:
        print(f"ERROR|Error fetching page {page}: {str(e)[:100]}", flush=True)
    return page_data

async def scrape_flipkart_reviews():
    asession = AsyncHTMLSession()
    base_url = (
        "https://www.flipkart.com/"
        "boat-450-pro-upto-70-hours-playback-bluetooth/product-reviews/"
        "itm575777beb2c09?pid=ACCGYUVXVVMZJRHF&lid=LSTACCGYUVXVVMZJRHFNV55IR"
        "&marketplace=FLIPKART&page="
    )
    total_pages = 10
    batch_size = 10
    delay_between_batches = 40
    all_data = []
    
    print(f"PROGRESS|Starting scraper: {total_pages} pages in {total_pages//batch_size} batches", flush=True)
    
    for batch_start in range(1, total_pages + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, total_pages)
        print(f"PROGRESS|Starting batch {batch_start}-{batch_end}...", flush=True)
        
        tasks = [fetch_page(asession, base_url, page) for page in range(batch_start, batch_end + 1)]
        batch_results = await asyncio.gather(*tasks)
        
        for page_data in batch_results:
            all_data.extend(page_data)
        
        # Calculate progress percentage
        progress_percent = int((batch_end / total_pages) * 100)
        print(f"PROGRESS|{progress_percent}%|{len(all_data)} reviews|Completed batch {batch_start}-{batch_end}", flush=True)
        
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
    # Support command line arguments for total_pages, batch_size, delay
    # Usage: python scraper.py [total_pages] [batch_size] [delay]
    asyncio.run(scrape_flipkart_reviews())
