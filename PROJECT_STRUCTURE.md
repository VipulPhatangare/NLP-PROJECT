# 📁 Project Structure

```
NLP Project/
│
├── 📄 package.json              # Node.js dependencies and scripts
├── 📄 requirements.txt          # Python dependencies
├── 📄 server.js                 # Express server (main backend)
├── 📄 config.py                 # Configuration settings
│
├── 📄 README.md                 # Detailed documentation
├── 📄 QUICKSTART.md            # Quick start guide
├── 📄 PROJECT_STRUCTURE.md     # This file
│
├── 🔧 install.ps1              # Installation script
├── 🔧 start.ps1                # Startup script
├── 📄 .gitignore               # Git ignore rules
│
├── 📂 scripts/                 # Python processing scripts
│   ├── scraper.py              # Web scraping (async)
│   ├── phase1_preprocessing.py # Text preprocessing pipeline
│   └── phase2_analysis.py      # NLP analysis pipeline
│
├── 📂 public/                  # Frontend files (served by Express)
│   ├── index.html              # Main dashboard UI
│   ├── styles.css              # Styling and layout
│   └── app.js                  # Frontend JavaScript logic
│
├── 📂 data/                    # Data storage (created at runtime)
│   ├── flipkart_boat_raw.csv   # Raw scraped reviews
│   └── flipkart_boat_cleaned.csv # Preprocessed reviews
│
└── 📂 results/                 # Analysis results (created at runtime)
    └── phase2_results.json     # Phase 2 analysis output
```

## 📝 File Descriptions

### Root Files

**package.json**
- Defines Node.js project metadata
- Lists Express, CORS, and other Node dependencies
- Contains npm scripts (start, dev)

**requirements.txt**
- Python package dependencies
- Includes: requests-html, beautifulsoup4, nltk, gensim, scikit-learn, pandas
- Used with: `pip install -r requirements.txt`

**server.js**
- Express web server
- REST API endpoints for data processing
- Serves static files from public/
- Routes: /api/status, /api/scrape, /api/phase1, /api/phase2, /api/data/*

**config.py**
- Centralized configuration
- Scraping settings (URLs, selectors, timing)
- NLP parameters (TF-IDF, LSA, Word2Vec)
- Customizable without editing main scripts

### Scripts Directory

**scripts/scraper.py**
- Asynchronous web scraping using requests-html
- Batch processing with configurable delays
- Extracts: reviews, ratings, titles, reviewer names
- Output: CSV file with raw data
- Usage: `python scripts/scraper.py [pages] [batch_size] [delay]`

**scripts/phase1_preprocessing.py**
- Emoji to word mapping
- Language detection (langdetect)
- Translation (deep-translator)
- Text cleaning and normalization
- Tokenization, stopword removal, lemmatization
- Output: Cleaned CSV + statistics JSON
- Usage: `python scripts/phase1_preprocessing.py [input] [output]`

**scripts/phase2_analysis.py**
- POS tagging (NLTK)
- Named Entity Recognition (rule-based)
- Bag-of-Words & TF-IDF vectorization
- LSA topic modeling
- Word2Vec embeddings and similarity
- Output: Comprehensive JSON results
- Usage: `python scripts/phase2_analysis.py [input] [output_dir]`

### Public Directory

**public/index.html**
- Single-page dashboard application
- Tabs: Overview, Phase 1, Phase 2, Data View
- Status indicators for each processing phase
- Control panel with action buttons
- Responsive layout with modern design

**public/styles.css**
- Modern, responsive styling
- CSS variables for theming
- Card-based layout
- Chart containers
- Table styling
- Mobile-friendly breakpoints

**public/app.js**
- Frontend application logic
- API communication with backend
- Chart.js integration for visualizations
- Dynamic content rendering
- Tab navigation
- Data table generation

### Data and Results

**data/**
- Created automatically by server
- Stores raw and processed CSV files
- Gitignored (except sample data)

**results/**
- Created automatically by server
- Stores JSON analysis results
- Gitignored

## 🔄 Data Flow

```
1. User clicks "Scrape Reviews"
   ↓
2. Frontend → POST /api/scrape → Backend
   ↓
3. Backend executes scripts/scraper.py
   ↓
4. Output: data/flipkart_boat_raw.csv
   ↓
5. User clicks "Run Phase 1"
   ↓
6. Backend executes scripts/phase1_preprocessing.py
   ↓
7. Output: data/flipkart_boat_cleaned.csv
   ↓
8. User clicks "Run Phase 2"
   ↓
9. Backend executes scripts/phase2_analysis.py
   ↓
10. Output: results/phase2_results.json
    ↓
11. Frontend fetches results via GET /api/results/phase2
    ↓
12. Charts and tables rendered with Chart.js
```

## 🎨 Frontend Architecture

```
index.html
  ├── Header (title, subtitle, author)
  ├── Status Bar (3 status indicators)
  ├── Control Panel (4 action buttons)
  ├── Loading Spinner
  └── Tabs Container
      ├── Overview Tab
      │   └── Stats Grid (4 cards)
      ├── Phase 1 Tab
      │   ├── Language Chart (Pie)
      │   ├── Stats Table
      │   └── Sample Reviews
      ├── Phase 2 Tab
      │   ├── POS Chart (Bar)
      │   ├── Adjectives Chart (Horizontal Bar)
      │   ├── Verbs Chart (Horizontal Bar)
      │   ├── NER Results (Grid)
      │   ├── Topics (Cards)
      │   └── Word2Vec Similarities (Grid)
      └── Data View Tab
          ├── Data Type Selector
          └── Data Table (Dynamic)
```

## 🔧 Backend API Architecture

```
Express Server (server.js)
  ├── Middleware
  │   ├── CORS
  │   ├── JSON Parser
  │   └── Static Files (public/)
  │
  └── Routes
      ├── GET /api/status
      │   └── Check if data files exist
      │
      ├── POST /api/scrape
      │   └── Run scraper.py
      │
      ├── POST /api/phase1
      │   └── Run phase1_preprocessing.py
      │
      ├── POST /api/phase2
      │   └── Run phase2_analysis.py
      │
      ├── GET /api/data/raw
      │   └── Return raw CSV as JSON
      │
      ├── GET /api/data/cleaned
      │   └── Return cleaned CSV as JSON
      │
      └── GET /api/results/phase2
          └── Return phase2_results.json
```

## 📊 Technology Stack

### Backend
- **Runtime**: Node.js 14+
- **Framework**: Express 4.x
- **Python**: 3.8+

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern styling, Flexbox, Grid
- **JavaScript**: ES6+, Async/await
- **Charts**: Chart.js 4.x

### Python Libraries
- **Web Scraping**: requests-html, beautifulsoup4
- **NLP**: nltk, gensim
- **ML**: scikit-learn
- **Data**: pandas, numpy
- **Translation**: deep-translator
- **Language Detection**: langdetect

## 🚀 Deployment Notes

### Development
```powershell
npm start
# Server runs on localhost:3000
```

### Production Considerations
1. Change to production-grade web server (PM2, Gunicorn)
2. Add authentication for API endpoints
3. Implement rate limiting
4. Add database for persistence
5. Use environment variables for config
6. Add logging (Winston, Morgan)
7. Implement error boundaries
8. Add data validation

## 📈 Performance Optimization

1. **Scraping**: Batch processing with delays
2. **Processing**: Stream large CSV files
3. **Frontend**: Lazy load charts
4. **Caching**: Cache analysis results
5. **CDN**: Use CDN for Chart.js

## 🔒 Security Considerations

1. Input validation on API endpoints
2. Sanitize file paths
3. Limit file upload sizes
4. Add CSRF protection
5. Use HTTPS in production
6. Validate Python script outputs
7. Sandbox Python execution

---

**Last Updated**: November 2025
**Version**: 1.0.0
**Author**: Vipul Phatangare
