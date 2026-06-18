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
    "product_path": "jbl-c50hi-wired/product-reviews/",
    "product_id": "itm7820dc2c9653e?pid=ACCFAMFQGCNEB8HM&lid=LSTACCFAMFQGCNEB8HMY6STV6&marketplace=FLIPKART"
}

# Scraping parameters
SCRAPING = {
    "total_pages": 30,          # Total number of pages to scrape
    "batch_size": 15,           # Pages per batch (to avoid rate limiting)
    "delay_between_batches": 80 # Seconds to wait between batches
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

# ===========================
# RAG SYSTEM CONFIGURATION
# ===========================

RAG = {
    # Document Processing
    "chunk_size": 300,              # Characters per chunk
    "chunk_overlap": 50,            # Overlapping characters between chunks
    
    # OpenAI Embeddings
    "embedding_model": "text-embedding-3-small",  # OpenAI model for embeddings
    "embedding_dimension": 1536,    # Dimension of text-embedding-3-small
    "embedding_batch_size": 100,    # Batch size for embedding generation
    
    # Pinecone Vector Store
    "pinecone_index_name": "nlp-project",
    "pinecone_metric": "cosine",    # Distance metric (cosine, euclidean, dotproduct)
    "pinecone_batch_size": 100,     # Batch size for vector upload
    
    # LLM Configuration
    "llm_provider": "openai",       # "openai" or "gemini"
    "openai_model": "gpt-4o-mini",  # OpenAI model for answer generation (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
    "gemini_model": "gemini-2.5-flash",  # Gemini model (fallback)
    "temperature": 0.1,             # Lower = more deterministic, higher = more creative
    "max_tokens": 1024,             # Maximum output tokens
    
    # Query Settings
    "top_k_results": 5,             # Number of similar reviews to retrieve
    "min_similarity_score": 0.7,    # Minimum similarity threshold
    "include_citations": True,      # Include review citations in answers
    
    # System Prompts
    "system_instruction": """You are a helpful AI assistant for analyzing product reviews. 
Use ONLY the provided review context to answer questions. 
If the context doesn't contain sufficient information, say "I don't have enough information in the reviews to answer that."
Provide concise, clear, and well-cited responses."""
}
