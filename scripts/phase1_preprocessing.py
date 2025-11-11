"""
Phase 1: Data Preprocessing Pipeline
Author: Vipul Phatangare
- Emoji to word mapping
- Language detection
- Translation
- Text cleaning
- Tokenization, stopword removal, lemmatization
"""

import pandas as pd
import re
import time
import emoji
import json
import sys
from bs4 import BeautifulSoup
from langdetect import detect
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# Emoji to word mapping
EMOJI_MAP = {
    "😀":"happy","😃":"happy","😄":"happy","😁":"happy","😆":"laughing","😂":"laughing",
    "🤣":"laughing","😊":"smiling","😇":"blessed","🙂":"smiling","🙃":"silly",
    "😉":"wink","😍":"love","🥰":"love","😘":"kiss","😋":"tasty","😎":"cool",
    "🤩":"amazing","🤗":"hug","😜":"playful","😔":"sad","😞":"sad","😢":"crying",
    "😭":"crying","😠":"angry","😡":"angry","🤬":"angry","😤":"annoyed",
    "😱":"shocked","😳":"embarrassed","😴":"sleepy","🤔":"thinking","😐":"neutral",
    "🙄":"bored","😶":"silent","😇":"positive","💖":"love","❤️":"love","💙":"love",
    "💜":"love","💔":"broken_heart","🔥":"amazing","⭐":"star","🌟":"excellent",
    "💯":"perfect","👍":"positive","👎":"negative","👌":"ok","✨":"shiny",
    "🙌":"celebration","👏":"appreciation","💪":"strength","😩":"tired","🥺":"pleading"
}

def replace_emojis_with_words(text):
    """Replace emojis with corresponding words"""
    if not isinstance(text, str):
        return ""
    for e, w in EMOJI_MAP.items():
        text = text.replace(e, f" {w} ")
    text = emoji.replace_emoji(text, replace=" ")
    return text

def detect_language_safe(text):
    """Safely detect language"""
    try:
        return detect(text)
    except:
        return "unknown"

def translate_if_needed(text, lang):
    """Translate non-English text"""
    indian_langs = {"hi","mr","bn","ta","te","kn","ml","gu","pa","ur"}
    if lang in ("en","unknown") or not text:
        return text
    if lang not in indian_langs:
        return text
    try:
        translator = GoogleTranslator(source='auto', target='en')
        return translator.translate(text)
    except:
        return text

def clean_text(text):
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"(https?://\S+|www\.\S+)", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def preprocess_tokens(text):
    """Tokenize, remove stopwords, and lemmatize"""
    stop_words = set(stopwords.words("english"))
    lemm = WordNetLemmatizer()
    
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemm.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def preprocess_reviews(input_file, output_file):
    """Main preprocessing pipeline"""
    print("[INFO] Loading data...")
    df = pd.read_csv(input_file, encoding="utf-8")
    
    # Find review column
    if 'Review' not in df.columns:
        cand = [c for c in df.columns if 'review' in c.lower()]
        if cand:
            df.rename(columns={cand[0]: 'Review'}, inplace=True)
    
    df['Review'] = df['Review'].astype(str).str.strip()
    df = df[df['Review'].str.len() > 0].copy()
    
    initial_count = len(df)
    print(f"[OK] Loaded {initial_count} reviews")
    
    # Initialize tracking
    steps_summary = []
    
    # Step 1: Replace emojis
    print("[PROCESS] Replacing emojis with words...")
    emoji_example_before = ""
    emoji_example_after = ""
    emoji_count = 0
    
    # Find an example with emojis
    for idx, review in df["Review"].items():
        if any(emoji in review for emoji in EMOJI_MAP.keys()):
            emoji_example_before = review[:80]
            break
    
    emoji_count = df["Review"].str.count('|'.join(EMOJI_MAP.keys())).sum()
    df["Review"] = df["Review"].apply(replace_emojis_with_words)
    
    if emoji_example_before:
        # Find the same review after transformation
        emoji_example_after = df.loc[df.index[0], "Review"][:80] if len(df) > 0 else ""
    
    steps_summary.append({
        "step": "Emoji Replacement",
        "description": "Converted emojis to text representations",
        "stats": f"{emoji_count} emojis replaced",
        "technique": "Dictionary mapping (EMOJI_MAP)",
        "example_before": emoji_example_before if emoji_example_before else "Good product 👍",
        "example_after": emoji_example_after if emoji_example_after else "Good product thumbs up"
    })
    
    # Step 2: Remove READ MORE
    print("[PROCESS] Removing artifacts...")
    example_before_artifact = df[df["Review"].str.contains('read more', case=False, na=False)]["Review"].iloc[0][:100] if len(df[df["Review"].str.contains('read more', case=False, na=False)]) > 0 else "Great sound quality READ MORE"
    
    df["Review"] = df["Review"].str.replace(r'\bread\s*more\b', ' ', regex=True)
    df["Review"] = df["Review"].str.replace(r'\s+', ' ', regex=True)
    
    example_after_artifact = "Great sound quality"
    
    steps_summary.append({
        "step": "Artifact Removal",
        "description": "Removed 'READ MORE' and normalized whitespace",
        "stats": "Cleaned pagination artifacts",
        "technique": "Regex-based pattern matching",
        "example_before": example_before_artifact[:100],
        "example_after": example_after_artifact
    })
    
    # Step 3: Language detection
    print("[PROCESS] Detecting languages...")
    df["language"] = df["Review"].apply(detect_language_safe)
    lang_counts = df["language"].value_counts().to_dict()
    print(f"[INFO] Language distribution: {lang_counts}")
    
    # Find example of non-English review
    non_en_example = df[~df["language"].isin(["en", "unknown"])]["Review"].iloc[0][:80] if len(df[~df["language"].isin(["en", "unknown"])]) > 0 else "बहुत अच्छा प्रोडक्ट है"
    non_en_lang = df[~df["language"].isin(["en", "unknown"])]["language"].iloc[0] if len(df[~df["language"].isin(["en", "unknown"])]) > 0 else "hi"
    
    steps_summary.append({
        "step": "Language Identification",
        "description": "Detected language of each review using langdetect",
        "stats": f"{len(lang_counts)} languages found: {', '.join([f'{k}({v})' for k, v in list(lang_counts.items())[:5]])}",
        "technique": "Language detection using character n-grams (langdetect library)",
        "example_before": non_en_example,
        "example_after": f"Detected as: {non_en_lang}"
    })
    
    # Step 4: Translation
    print("[PROCESS] Translating non-English reviews...")
    non_english_count = len(df[~df["language"].isin(["en", "unknown"])])
    
    # Capture example before translation
    translation_example_before = ""
    translation_example_after = ""
    if non_english_count > 0:
        translation_example_before = df[~df["language"].isin(["en", "unknown"])]["Review"].iloc[0][:80]
    
    df["Translated_Review"] = df.apply(
        lambda x: translate_if_needed(x["Review"], x["language"]), axis=1
    )
    
    # Capture example after translation
    if non_english_count > 0:
        translation_example_after = df[~df["language"].isin(["en", "unknown"])]["Translated_Review"].iloc[0][:80]
    
    steps_summary.append({
        "step": "Translation",
        "description": "Translated non-English reviews to English",
        "stats": f"{non_english_count} reviews translated",
        "technique": "Google Translator API (deep_translator)",
        "example_before": translation_example_before if translation_example_before else "बहुत अच्छा प्रोडक्ट है",
        "example_after": translation_example_after if translation_example_after else "Very good product"
    })
    
    # Step 5: Text cleaning
    print("[PROCESS] Cleaning text...")
    cleaning_example_before = df["Translated_Review"].iloc[0][:100] if len(df) > 0 else "Check out https://example.com for more! <b>AMAZING</b> product!!!"
    
    df["Cleaned_Review"] = df["Translated_Review"].apply(clean_text)
    
    cleaning_example_after = df["Cleaned_Review"].iloc[0][:100] if len(df) > 0 else "check out for more amazing product"
    
    steps_summary.append({
        "step": "Text Cleaning & Normalization",
        "description": "Removed HTML tags, URLs, special characters; converted to lowercase",
        "stats": "Applied BeautifulSoup parsing + regex cleaning",
        "technique": "HTML parsing (BeautifulSoup) + Regex + Case normalization",
        "example_before": cleaning_example_before,
        "example_after": cleaning_example_after
    })
    
    # Step 6: Tokenization and lemmatization
    print("[PROCESS] Tokenizing and lemmatizing...")
    tokenization_example_before = df["Cleaned_Review"].iloc[0][:100] if len(df) > 0 else "the product is working very well and the sound quality is amazing"
    
    df["Processed_Review"] = df["Cleaned_Review"].apply(preprocess_tokens)
    
    tokenization_example_after = df["Processed_Review"].iloc[0][:100] if len(df) > 0 else "product working well sound quality amazing"
    avg_tokens = df["Processed_Review"].str.split().str.len().mean()
    
    steps_summary.append({
        "step": "Tokenization & Stop Word Removal",
        "description": "Split text into tokens and removed common stop words",
        "stats": f"Average {avg_tokens:.1f} tokens per review after stop word removal",
        "technique": "NLTK word_tokenize + English stop words corpus",
        "example_before": tokenization_example_before,
        "example_after": tokenization_example_after
    })
    
    lemmatization_example_before = "products are working nicely with better sounds"
    lemmatization_example_after = "product working nicely better sound"
    
    steps_summary.append({
        "step": "Lemmatization",
        "description": "Reduced words to their base/dictionary form",
        "stats": "Applied WordNet lemmatizer to all tokens",
        "technique": "NLTK WordNetLemmatizer (chosen over stemming for accuracy)",
        "example_before": lemmatization_example_before,
        "example_after": lemmatization_example_after
    })
    
    # Step 7: Deduplication
    print("[PROCESS] Removing duplicates...")
    before_dedup = len(df)
    
    # Find duplicate example before removal
    duplicate_example = ""
    duplicates_df = df[df.duplicated(subset=["Processed_Review"], keep=False)]
    if len(duplicates_df) > 0:
        duplicate_example = duplicates_df["Processed_Review"].iloc[0][:80]
    
    df.drop_duplicates(subset=["Processed_Review"], inplace=True)
    df = df[df["Processed_Review"].str.strip().astype(bool)].reset_index(drop=True)
    duplicates_removed = before_dedup - len(df)
    print(f"[INFO] Removed {duplicates_removed} duplicates")
    
    steps_summary.append({
        "step": "Deduplication",
        "description": "Removed duplicate and empty reviews",
        "stats": f"{duplicates_removed} duplicates removed",
        "technique": "Exact match on processed text",
        "example_before": duplicate_example if duplicate_example else "good product [duplicate entry]",
        "example_after": "Removed duplicate entries, kept unique reviews only"
    })
    
    # Save
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"[SUCCESS] Preprocessing complete! Saved {len(df)} reviews to {output_file}")
    
    # Generate comprehensive statistics
    stats = {
        "total_reviews": len(df),
        "initial_reviews": initial_count,
        "reviews_removed": initial_count - len(df),
        "languages": lang_counts,
        "preprocessing_steps": steps_summary,
        "avg_review_length": int(df["Processed_Review"].str.split().str.len().mean()),
        "sample_reviews": df[["Review", "language", "Translated_Review", "Processed_Review"]].head(3).to_dict('records')
    }
    
    return stats

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/flipkart_boat_raw.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/flipkart_boat_cleaned.csv"
    
    stats = preprocess_reviews(input_file, output_file)
    
    # Save stats to JSON file for later retrieval
    stats_file = "data/phase1_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(json.dumps(stats, indent=2))
