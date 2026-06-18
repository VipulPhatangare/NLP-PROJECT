# NLP Review Analysis Dashboard

A web-based system to scrape, preprocess, and analyze product reviews from Flipkart using NLP techniques.

## Features

- Scrapes product reviews from Flipkart (multi-page)
- Translates Indian language reviews to English
- NLP preprocessing: tokenization, stopword removal, lemmatization
- Sentiment analysis, topic modeling, and advanced text analytics
- Interactive web dashboard to visualize results

## Tech Stack

- **Backend:** Node.js + Express
- **Frontend:** HTML, CSS, JavaScript
- **NLP/ML:** Python (NLTK, scikit-learn, gensim, pandas)
- **Scraping:** BeautifulSoup, requests-html

## Project Structure

```
├── server.js              # Express server
├── config.py              # Scraping & NLP configuration
├── requirements.txt       # Python dependencies
├── package.json           # Node.js dependencies
├── start.ps1              # One-click startup script (Windows)
├── public/                # Frontend (HTML, CSS, JS)
├── scripts/
│   ├── scraper.py         # Flipkart review scraper
│   ├── phase1_preprocessing.py  # Cleaning & translation
│   ├── phase2_analysis.py       # Sentiment & basic NLP
│   └── phase3_advanced.py       # Advanced analytics
├── data/                  # Scraped & processed data
└── results/               # Analysis output & charts
```

## Setup & Run

### Prerequisites
- Node.js (v16+)
- Python 3.8+

### Start the app (Windows)

```powershell
.\start.ps1
```

This will automatically:
1. Install Node.js dependencies
2. Create and activate Python virtual environment
3. Install Python dependencies
4. Download required NLTK data
5. Start the server at `http://localhost:3000`

### Manual setup

```bash
# Install Node dependencies
npm install

# Create Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
source venv/bin/activate       # Linux/Mac

# Install Python dependencies
pip install -r requirements.txt

# Start server
npm start
```

## Configuration

Edit [config.py](config.py) to change:
- **Product URL** — the Flipkart product to scrape
- **Scraping parameters** — number of pages, batch size, delays
- **NLP settings** — languages, translation options

## Usage

1. Open `http://localhost:3000` in your browser
2. Use the dashboard to trigger scraping, preprocessing, and analysis phases
3. View results and visualizations directly in the browser

## Author

Vipul Phatangare
