"""
Configuration file for NLP Review Analysis
Modify these settings as needed
"""

# ===========================
# SCRAPING CONFIGURATION
# ===========================

# Flipkart product URL components
PRODUCT_URL = {
    "base": "https://www.flipkart.com/",
    "product_path": "boat-450-pro-upto-70-hours-playback-bluetooth/product-reviews/",
    "product_id": "itm575777beb2c09?pid=ACCGYUVXVVMZJRHF&lid=LSTACCGYUVXVVMZJRHFNV55IR&marketplace=FLIPKART"
}

# Scraping parameters
SCRAPING = {
    "total_pages": 60,          # Total number of pages to scrape
    "batch_size": 15,           # Pages per batch (to avoid rate limiting)
    "delay_between_batches": 60, # Seconds to wait between batches
    "render_timeout": 40,        # Timeout for page rendering
    "render_sleep": 2            # Sleep after rendering
}

# CSS selectors for Flipkart
SELECTORS = {
    "reviews": "div.ZmyHeo",
    "ratings": "div.XQDdHH",
    "titles": "p.z9E0IG",
    "names": "p._2NsDsF.AwS1CA"
}

# ===========================
# PREPROCESSING CONFIGURATION
# ===========================

# Languages to translate
INDIAN_LANGUAGES = {"hi", "mr", "bn", "ta", "te", "kn", "ml", "gu", "pa", "ur"}

# Translation settings
TRANSLATION = {
    "source": "auto",
    "target": "en",
    "delay": 0.05  # Delay between translation calls (seconds)
}

# Text cleaning patterns
CLEANING = {
    "remove_urls": True,
    "remove_html": True,
    "remove_special_chars": True,
    "lowercase": True,
    "remove_stopwords": True,
    "lemmatize": True
}

# ===========================
# ANALYSIS CONFIGURATION
# ===========================

# POS Tagging
POS_TAGS = {
    "adjectives": {"JJ", "JJR", "JJS"},
    "verbs": {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"},
    "nouns": {"NN", "NNS", "NNP", "NNPS"}
}

# Named Entity Recognition patterns
NER_PATTERNS = {
    "brands": [r"\bboat\b", r"\bboAt\b", r"\bbo-at\b"],
    "features": [
        "battery", "backup", "bass", "sound", "mic", "microphone",
        "call", "bluetooth", "charging", "build", "comfort", "design",
        "quality", "noise", "cancellation", "latency", "gaming",
        "voice", "volume", "pairing", "range", "price", "warranty"
    ],
    "duration": r"\b(\d+)\s*(hr|hrs|hour|hours|h|mins?|minutes?)\b",
    "version": r"\b(v?\d+(?:\.\d+)?)\b",
    "price": r"(?:₹|rs\.?\s*)\s*\d+(?:,\d{3})*"
}

# TF-IDF parameters
TFIDF = {
    "min_df": 3,      # Minimum document frequency
    "max_df": 0.9,    # Maximum document frequency
    "max_features": None
}

# LSA Topic Modeling
LSA = {
    "n_topics": 5,     # Number of topics to extract
    "n_top_words": 12  # Top words per topic
}

# Word2Vec parameters
WORD2VEC = {
    "vector_size": 100,
    "window": 5,
    "min_count": 3,
    "workers": 4,
    "sg": 0,           # 0=CBOW, 1=Skip-gram
    "epochs": 15
}

# Seed terms for similarity analysis
SEED_TERMS = [
    "battery", "bass", "mic", "comfort", "design",
    "quality", "charging", "noise", "warranty", "price"
]

# ===========================
# FILE PATHS
# ===========================

PATHS = {
    "data_dir": "data",
    "results_dir": "results",
    "raw_data": "data/flipkart_boat_raw.csv",
    "cleaned_data": "data/flipkart_boat_cleaned.csv",
    "phase2_results": "results/phase2_results.json"
}

# ===========================
# SERVER CONFIGURATION
# ===========================

SERVER = {
    "port": 3000,
    "host": "localhost",
    "cors_enabled": True
}

# ===========================
# VISUALIZATION
# ===========================

VISUALIZATION = {
    "chart_colors": [
        "#4f46e5", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
        "#ec4899", "#14b8a6", "#f97316", "#06b6d4", "#84cc16"
    ],
    "max_chart_items": 15,  # Maximum items to display in charts
    "sample_reviews": 5     # Number of sample reviews to display
}
