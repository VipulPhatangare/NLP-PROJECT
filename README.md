# 🎯 NLP Product Review Analysis Dashboard

A comprehensive web-based system for scraping, analyzing, and visualizing product reviews using classical NLP techniques.

**Author:** Vipul Phatangare

## 📋 Features

- **Web Scraping**: Automated Flipkart review scraping with batch processing
- **Phase 1 - Preprocessing**:
  - Emoji to word mapping
  - Language detection
  - Automatic translation
  - Text cleaning and normalization
  - Tokenization, stopword removal, lemmatization
  
- **Phase 2 - Analysis**:
  - POS (Part-of-Speech) tagging
  - Named Entity Recognition (rule-based)
  - Bag of Words & TF-IDF representations
  - LSA Topic Modeling
  - Word2Vec semantic similarities
  
- **Interactive Dashboard**:
  - Beautiful responsive UI
  - Real-time charts and graphs
  - Phase-wise result visualization
  - Complete data viewing and export

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- npm or yarn

### Step 1: Clone/Setup Project

```powershell
cd "c:\Users\vipul\OneDrive\Desktop\web dev\Collage projects\NLP Project"
```

### Step 2: Install Node Dependencies

```powershell
npm install
```

### Step 3: Install Python Dependencies

```powershell
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate

# Install requirements
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

## 🎮 Usage

### Start the Server

```powershell
npm start
```

The server will start at `http://localhost:3000`

### Using the Dashboard

1. **Open Browser**: Navigate to `http://localhost:3000`

2. **Scrape Reviews**:
   - Click "Scrape Reviews" button
   - Wait for the scraping process to complete (may take 15-30 minutes)
   - Reviews will be saved to `data/flipkart_boat_raw.csv`

3. **Run Phase 1 (Preprocessing)**:
   - Click "Run Phase 1" button
   - Preprocessing will clean and translate reviews
   - Results saved to `data/flipkart_boat_cleaned.csv`
   - View language distribution and sample reviews

4. **Run Phase 2 (Analysis)**:
   - Click "Run Phase 2" button
   - Comprehensive NLP analysis will be performed
   - Results saved to `results/phase2_results.json`
   - View POS tags, entities, topics, and word similarities

5. **View All Data**:
   - Click "View All Data" button
   - Select data type (Raw or Cleaned)
   - Browse complete dataset in table format
   - Export as CSV

## 📁 Project Structure

```
NLP Project/
├── server.js              # Express server
├── package.json           # Node dependencies
├── requirements.txt       # Python dependencies
├── scripts/
│   ├── scraper.py        # Web scraping script
│   ├── phase1_preprocessing.py  # Preprocessing pipeline
│   └── phase2_analysis.py      # Analysis pipeline
├── public/
│   ├── index.html        # Dashboard UI
│   ├── styles.css        # Styling
│   └── app.js            # Frontend JavaScript
├── data/                 # Scraped and processed data
└── results/              # Analysis results
```

## 🔧 API Endpoints

- `GET /api/status` - Check data availability
- `POST /api/scrape` - Start scraping reviews
- `POST /api/phase1` - Run preprocessing
- `POST /api/phase2` - Run analysis
- `GET /api/data/raw` - Get raw reviews
- `GET /api/data/cleaned` - Get cleaned reviews
- `GET /api/results/phase2` - Get analysis results

## 📊 Analysis Outputs

### Phase 1 Results
- Language distribution chart
- Preprocessing statistics table
- Sample processed reviews

### Phase 2 Results
- **POS Distribution**: Bar chart of part-of-speech tags
- **Top Words**: Most frequent adjectives and verbs
- **Named Entities**: Brands, features, durations, versions, prices
- **Topics**: LSA-based topic modeling with top keywords
- **Semantic Similarity**: Word2Vec similarities for key terms

## 🎨 Dashboard Features

- **Real-time Status**: Track completion of each phase
- **Interactive Charts**: Powered by Chart.js
- **Responsive Design**: Works on desktop and mobile
- **Data Export**: Download CSV files
- **Phase Navigation**: Easy tab-based navigation

## 🛠️ Technologies Used

### Backend
- **Node.js** with Express
- **Python 3.x**

### Python Libraries
- `requests-html` - Web scraping
- `beautifulsoup4` - HTML parsing
- `deep-translator` - Translation
- `langdetect` - Language detection
- `nltk` - NLP toolkit
- `gensim` - Word2Vec
- `scikit-learn` - ML algorithms
- `pandas` - Data manipulation

### Frontend
- **HTML5/CSS3**
- **Vanilla JavaScript**
- **Chart.js** - Data visualization

## ⚠️ Important Notes

1. **Scraping Duration**: Scraping 60 pages takes 15-30 minutes due to rate limiting
2. **Python Environment**: Always activate virtual environment before running scripts
3. **Browser Compatibility**: Best viewed in Chrome, Firefox, or Edge
4. **Data Persistence**: All data is stored locally in `data/` and `results/` folders

## 🐛 Troubleshooting

### Import Errors
```powershell
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

### NLTK Data Missing
```powershell
python -c "import nltk; nltk.download('all')"
```

### Port Already in Use
Edit `server.js` and change `PORT = 3000` to another port

### Scraping Fails
- Check internet connection
- Flipkart might have changed HTML structure
- Try reducing `batchSize` in scraper settings

## 📝 License

This project is for educational purposes as part of NLP coursework.

## 👨‍💻 Author

**Vipul Phatangare**

---

## 🚀 Quick Start Commands

```powershell
# Install everything
npm install
pip install -r requirements.txt

# Start server
npm start

# Open browser
start http://localhost:3000
```

Enjoy analyzing reviews! 🎉
