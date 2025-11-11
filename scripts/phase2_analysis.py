"""
Phase 2: Syntactic & Semantic Analysis
Author: Vipul Phatangare
- POS Tagging
- NER (Rule-based)
- BoW & TF-IDF
- Word2Vec
- LSA Topic Modeling
- Sentiment Analysis (Lexicon-based + LSTM)
"""

import pandas as pd
import numpy as np
import re
import json
import sys
from collections import Counter, defaultdict

import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# TensorFlow/Keras for LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

# Download NLTK data
try:
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

def pos_tagging_analysis(df, text_col):
    """Perform POS tagging analysis"""
    print("[POS] Performing POS tagging...")
    
    def pos_tag_doc(doc):
        tokens = word_tokenize(str(doc))
        return pos_tag(tokens)
    
    df["pos_tags"] = df[text_col].apply(pos_tag_doc)
    
    # Aggregate POS distribution
    pos_counts = Counter()
    for tags in df["pos_tags"]:
        pos_counts.update([t[1] for t in tags])
    
    pos_dist = pd.DataFrame(
        sorted(pos_counts.items(), key=lambda x: x[1], reverse=True),
        columns=["POS_Tag", "Count"]
    )
    
    # Extract adjectives and verbs
    def keep_tags(tags, allowed):
        return [w for (w, t) in tags if t in allowed]
    
    df["adjectives"] = df["pos_tags"].apply(lambda ts: keep_tags(ts, {"JJ","JJR","JJS"}))
    df["verbs"] = df["pos_tags"].apply(lambda ts: keep_tags(ts, {"VB","VBD","VBG","VBN","VBP","VBZ"}))
    
    # Most common adjectives and verbs
    all_adj = [adj for adjs in df["adjectives"] for adj in adjs]
    all_verbs = [verb for verbs in df["verbs"] for verb in verbs]
    
    top_adj = Counter(all_adj).most_common(20)
    top_verbs = Counter(all_verbs).most_common(20)
    
    return {
        "pos_distribution": pos_dist.to_dict('records'),
        "top_adjectives": top_adj,
        "top_verbs": top_verbs
    }

def rule_based_ner(df, text_col):
    """Extract entities using rule-based patterns"""
    print("[NER] Performing Named Entity Recognition...")
    
    brand_patterns = [
        r"\bboat\b", r"\bboAt\b", r"\bbo-at\b"
    ]
    feature_keywords = [
        "battery","backup","bass","sound","mic","microphone","call","bluetooth",
        "charging","build","comfort","design","quality","noise","cancellation",
        "latency","gaming","voice","volume","pairing","range","price","warranty"
    ]
    duration_pattern = r"\b(\d+)\s*(hr|hrs|hour|hours|h|mins?|minutes?)\b"
    version_pattern = r"\b(v?\d+(?:\.\d+)?)\b"
    price_pattern = r"(?:₹|rs\.?\s*)\s*\d+(?:,\d{3})*"
    
    def extract_entities(text):
        ents = defaultdict(list)
        t = " " + str(text) + " "
        
        # Brands
        for pat in brand_patterns:
            for m in re.finditer(pat, t, flags=re.IGNORECASE):
                ents["BRAND"].append(m.group(0).strip().lower())
        
        # Features
        for feat in feature_keywords:
            if re.search(rf"\b{re.escape(feat)}\b", t, flags=re.IGNORECASE):
                ents["FEATURE"].append(feat)
        
        # Duration
        for m in re.finditer(duration_pattern, t, flags=re.IGNORECASE):
            ents["DURATION"].append(m.group(0).lower())
        
        # Version
        for m in re.finditer(version_pattern, t, flags=re.IGNORECASE):
            ents["VERSION"].append(m.group(0).lower())
        
        # Price
        for m in re.finditer(price_pattern, t, flags=re.IGNORECASE):
            ents["PRICE"].append(m.group(0))
        
        # Dedup
        for k in ents:
            ents[k] = sorted(set(ents[k]))
        return dict(ents)
    
    df["entities"] = df[text_col].apply(extract_entities)
    
    # Flatten for analysis
    ner_rows = []
    for idx, ents in enumerate(df["entities"]):
        for typ, vals in ents.items():
            for v in vals:
                ner_rows.append({"doc_id": idx, "type": typ, "value": v})
    
    ner_df = pd.DataFrame(ner_rows)
    
    if not ner_df.empty:
        freq = ner_df.groupby(["type","value"]).size().reset_index(name="count")
        freq = freq.sort_values("count", ascending=False)
        top_entities = freq.head(30).to_dict('records')
        
        # By type
        entity_stats = {}
        for entity_type in ner_df["type"].unique():
            type_data = ner_df[ner_df["type"] == entity_type]
            entity_stats[entity_type] = {
                "count": len(type_data),
                "unique": type_data["value"].nunique(),
                "top_values": type_data["value"].value_counts().head(10).to_dict()
            }
    else:
        top_entities = []
        entity_stats = {}
    
    return {
        "top_entities": top_entities,
        "entity_stats": entity_stats
    }

def bow_tfidf_analysis(docs):
    """Create BoW and TF-IDF representations"""
    print("[BOW] Creating BoW and TF-IDF representations...")
    
    bow_vec = CountVectorizer(min_df=3, max_df=0.9)
    X_bow = bow_vec.fit_transform(docs)
    
    tfidf_vec = TfidfVectorizer(min_df=3, max_df=0.9)
    X_tfidf = tfidf_vec.fit_transform(docs)
    
    # Top terms
    feature_names = tfidf_vec.get_feature_names_out()
    
    def top_tfidf_terms(row_vector, feature_names, topn=10):
        coo = row_vector.tocoo()
        pairs = sorted(zip(coo.col, coo.data), key=lambda x: x[1], reverse=True)[:topn]
        return [(feature_names[c], round(w,4)) for c, w in pairs]
    
    sample_top_terms = []
    for i in range(min(5, X_tfidf.shape[0])):
        sample_top_terms.append({
            "doc_id": i,
            "top_terms": top_tfidf_terms(X_tfidf[i], feature_names, topn=10)
        })
    
    return {
        "bow_shape": X_bow.shape,
        "tfidf_shape": X_tfidf.shape,
        "vocab_size": len(feature_names),
        "sample_top_terms": sample_top_terms
    }, X_tfidf, tfidf_vec

def lsa_topic_modeling(X_tfidf, tfidf_vec, n_topics=5):
    """Perform LSA topic modeling"""
    print("[LSA] Performing LSA topic modeling...")
    
    lsa = TruncatedSVD(n_components=n_topics, random_state=42)
    lsa_mat = lsa.fit_transform(X_tfidf)
    
    terms = tfidf_vec.get_feature_names_out()
    topic_terms = []
    
    for topic_idx, comp in enumerate(lsa.components_):
        top_indices = np.argsort(comp)[::-1][:12]
        top_words = [terms[i] for i in top_indices]
        topic_terms.append({
            "topic": f"Topic {topic_idx+1}",
            "top_words": top_words,
            "variance_explained": float(lsa.explained_variance_ratio_[topic_idx])
        })
    
    return {
        "topics": topic_terms,
        "total_variance_explained": float(sum(lsa.explained_variance_ratio_))
    }

def word2vec_analysis(docs):
    """Train Word2Vec and find similar words"""
    print("[W2V] Training Word2Vec model...")
    
    sentences = [simple_preprocess(doc, deacc=True, min_len=2) for doc in docs if isinstance(doc, str)]
    
    w2v = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=3,
        workers=4,
        sg=0,
        epochs=15
    )
    
    seed_terms = ["battery","bass","mic","comfort","design","quality","charging","noise","warranty","price"]
    similar_map = {}
    
    for term in seed_terms:
        if term in w2v.wv.key_to_index:
            similar = w2v.wv.most_similar(term, topn=5)
            similar_map[term] = [(w, float(s)) for (w, s) in similar]
        else:
            similar_map[term] = []
    
    return similar_map

def lexicon_based_sentiment(df, text_col):
    """
    Lexicon-based sentiment analysis using VADER
    """
    print("[SENTIMENT] Performing lexicon-based sentiment analysis (VADER)...")
    
    # Initialize VADER
    sia = SentimentIntensityAnalyzer()
    
    # Analyze sentiment for each review
    sentiments = []
    for text in df[text_col]:
        scores = sia.polarity_scores(str(text))
        sentiments.append(scores)
    
    df['vader_neg'] = [s['neg'] for s in sentiments]
    df['vader_neu'] = [s['neu'] for s in sentiments]
    df['vader_pos'] = [s['pos'] for s in sentiments]
    df['vader_compound'] = [s['compound'] for s in sentiments]
    
    # Classify sentiment based on compound score
    def classify_sentiment(compound):
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    df['vader_sentiment'] = df['vader_compound'].apply(classify_sentiment)
    
    # Calculate distribution
    sentiment_dist = df['vader_sentiment'].value_counts().to_dict()
    
    # Calculate average scores
    avg_scores = {
        'avg_compound': float(df['vader_compound'].mean()),
        'avg_positive': float(df['vader_pos'].mean()),
        'avg_negative': float(df['vader_neg'].mean()),
        'avg_neutral': float(df['vader_neu'].mean())
    }
    
    print(f"[OK] Sentiment distribution: {sentiment_dist}")
    
    return {
        'method': 'VADER Lexicon-based',
        'sentiment_distribution': sentiment_dist,
        'average_scores': avg_scores,
        'total_reviews': len(df)
    }

def identify_key_sentiment_phrases(df, original_col, n_phrases=5):
    """
    Identify key phrases contributing to positive and negative sentiment
    """
    print("[SENTIMENT] Identifying key sentiment phrases...")
    
    sia = SentimentIntensityAnalyzer()
    
    # Get most positive and negative reviews
    positive_reviews = df.nlargest(n_phrases, 'vader_compound')[original_col].tolist()
    negative_reviews = df.nsmallest(n_phrases, 'vader_compound')[original_col].tolist()
    
    key_phrases = {
        'most_positive_phrases': [
            {
                'text': str(text)[:200],
                'score': float(df[df[original_col] == text]['vader_compound'].iloc[0]) if len(df[df[original_col] == text]) > 0 else 0.0
            }
            for text in positive_reviews
        ],
        'most_negative_phrases': [
            {
                'text': str(text)[:200],
                'score': float(df[df[original_col] == text]['vader_compound'].iloc[0]) if len(df[df[original_col] == text]) > 0 else 0.0
            }
            for text in negative_reviews
        ]
    }
    
    return key_phrases

def lstm_sentiment_analysis(df, text_col, original_col):
    """
    LSTM-based sentiment analysis
    Creates a simple LSTM model for sentiment classification
    """
    print("[SENTIMENT] Building LSTM sentiment model...")
    
    # Prepare data
    texts = df[original_col].astype(str).tolist()
    
    # Use VADER labels as training labels
    labels = df['vader_sentiment'].map({'positive': 2, 'neutral': 1, 'negative': 0}).values
    
    # Tokenization
    max_words = 5000
    max_len = 100
    
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    # Convert labels to categorical
    labels_categorical = tf.keras.utils.to_categorical(labels, num_classes=3)
    
    # Build LSTM model
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("[LSTM] Training model...")
    # Train with small epochs for quick analysis
    history = model.fit(
        padded, labels_categorical,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Make predictions
    predictions = model.predict(padded, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Map back to sentiment labels
    class_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    df['lstm_sentiment'] = [class_map[c] for c in predicted_classes]
    df['lstm_confidence'] = [float(np.max(pred)) for pred in predictions]
    
    # Calculate distribution
    lstm_dist = df['lstm_sentiment'].value_counts().to_dict()
    
    # Calculate accuracy compared to VADER
    vader_labels = df['vader_sentiment'].values
    lstm_labels = df['lstm_sentiment'].values
    agreement = np.mean(vader_labels == lstm_labels)
    
    print(f"[OK] LSTM sentiment distribution: {lstm_dist}")
    print(f"[OK] Agreement with VADER: {agreement:.2%}")
    
    return {
        'method': 'Bidirectional LSTM with Embeddings',
        'architecture': 'Embedding(128) -> BiLSTM(64) -> BiLSTM(32) -> Dense(64) -> Dense(3)',
        'training_epochs': 5,
        'sentiment_distribution': lstm_dist,
        'avg_confidence': float(df['lstm_confidence'].mean()),
        'agreement_with_vader': float(agreement),
        'final_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1])
    }

def analyze_reviews(input_file, output_dir):
    """Main analysis pipeline"""
    print(f"[LOAD] Loading cleaned data from {input_file}...")
    df = pd.read_csv(input_file, encoding="utf-8")
    
    text_col = "Processed_Review"
    assert text_col in df.columns, f"Column '{text_col}' not found!"
    
    # Check for original review column
    original_col = "Translated_Review" if "Translated_Review" in df.columns else "Review"
    
    docs = df[text_col].astype(str).tolist()
    print(f"[OK] Loaded {len(docs)} reviews")
    
    results = {}
    
    # POS Tagging
    results["pos_analysis"] = pos_tagging_analysis(df, text_col)
    
    # NER
    results["ner_analysis"] = rule_based_ner(df, text_col)
    
    # BoW & TF-IDF
    bow_tfidf_result, X_tfidf, tfidf_vec = bow_tfidf_analysis(docs)
    results["bow_tfidf"] = bow_tfidf_result
    
    # LSA Topics
    results["lsa_topics"] = lsa_topic_modeling(X_tfidf, tfidf_vec, n_topics=5)
    
    # Word2Vec
    results["word2vec_similarities"] = word2vec_analysis(docs)
    
    # Sentiment Analysis - Lexicon-based (VADER)
    results["sentiment_lexicon"] = lexicon_based_sentiment(df, original_col)
    
    # Identify key sentiment phrases
    results["key_sentiment_phrases"] = identify_key_sentiment_phrases(df, original_col, n_phrases=5)
    
    # Sentiment Analysis - LSTM
    results["sentiment_lstm"] = lstm_sentiment_analysis(df, text_col, original_col)
    
    # Overall sentiment score
    positive_count = results["sentiment_lexicon"]["sentiment_distribution"].get("positive", 0)
    negative_count = results["sentiment_lexicon"]["sentiment_distribution"].get("negative", 0)
    neutral_count = results["sentiment_lexicon"]["sentiment_distribution"].get("neutral", 0)
    total = positive_count + negative_count + neutral_count
    
    results["overall_sentiment"] = {
        "positive_percentage": round((positive_count / total * 100), 2) if total > 0 else 0,
        "negative_percentage": round((negative_count / total * 100), 2) if total > 0 else 0,
        "neutral_percentage": round((neutral_count / total * 100), 2) if total > 0 else 0,
        "overall_score": results["sentiment_lexicon"]["average_scores"]["avg_compound"],
        "verdict": "Positive" if results["sentiment_lexicon"]["average_scores"]["avg_compound"] > 0.05 else ("Negative" if results["sentiment_lexicon"]["average_scores"]["avg_compound"] < -0.05 else "Neutral")
    }
    
    # Save results
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/phase2_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[SUCCESS] Phase 2 analysis complete! Results saved to {output_dir}/phase2_results.json")
    
    return results

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/flipkart_boat_cleaned.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    
    results = analyze_reviews(input_file, output_dir)
    print(json.dumps(results, indent=2))
