import pandas as pd
import numpy as np
import json
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import Counter
import re

def load_data():
    """Load cleaned reviews and Phase 2 results"""
    try:
        df = pd.read_csv("data/flipkart_boat_cleaned.csv", encoding="utf-8")
        with open("results/phase2_results.json", "r", encoding="utf-8") as f:
            phase2_data = json.load(f)
        return df, phase2_data
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)

def extract_representative_reviews(df, n_clusters=5, top_reviews=5):
    """
    Extract representative reviews using clustering and similarity analysis
    """
    print("[PROCESS] Extracting representative reviews...")
    
    # Create TF-IDF vectors for all reviews
    vectorizer = TfidfVectorizer(max_features=500, min_df=2, max_df=0.8)
    tfidf_matrix = vectorizer.fit_transform(df["Processed_Review"])
    
    print(f"[INFO] Created TF-IDF matrix with shape {tfidf_matrix.shape}")
    
    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"[INFO] Calculated similarity matrix")
    
    # Perform clustering
    n_clusters = min(n_clusters, len(df) // 10)  # Ensure reasonable cluster count
    if n_clusters < 2:
        n_clusters = 2
        
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(tfidf_matrix)
    
    print(f"[INFO] Created {n_clusters} clusters")
    
    # For each cluster, find representative reviews
    representative_reviews = []
    
    for cluster_id in range(n_clusters):
        cluster_indices = df[df["cluster"] == cluster_id].index.tolist()
        cluster_size = len(cluster_indices)
        
        if cluster_size == 0:
            continue
            
        # Get similarity scores within cluster
        cluster_similarities = similarity_matrix[cluster_indices][:, cluster_indices]
        
        # Find review with highest average similarity to others in cluster (centroid)
        avg_similarities = cluster_similarities.mean(axis=1)
        best_idx = cluster_indices[np.argmax(avg_similarities)]
        
        representative_reviews.append({
            "cluster_id": int(cluster_id),
            "cluster_size": cluster_size,
            "review_original": df.loc[best_idx, "Review"],
            "review_processed": df.loc[best_idx, "Processed_Review"],
            "similarity_score": float(np.max(avg_similarities)),
            "represents": f"{cluster_size} similar reviews"
        })
    
    # Sort by cluster size (most common feedback first)
    representative_reviews.sort(key=lambda x: x["cluster_size"], reverse=True)
    
    print(f"[SUCCESS] Extracted {len(representative_reviews)} representative reviews")
    
    return representative_reviews[:top_reviews], similarity_matrix, df

def extract_key_topics(df, phase2_data):
    """Extract key topics and features from reviews"""
    topics = {}
    
    # From TF-IDF
    if "tfidf_top_terms" in phase2_data:
        tfidf_terms = [term for term, _ in phase2_data["tfidf_top_terms"][:20]]
        topics["important_terms"] = tfidf_terms
    
    # From entities
    if "entities" in phase2_data:
        entity_list = []
        for entity_type, entities in phase2_data["entities"].items():
            entity_list.extend([e for e, _ in entities[:5]])
        topics["mentioned_entities"] = entity_list
    
    return topics

def generate_qa_pairs(df, phase2_data, representative_reviews):
    """
    Generate insightful Q&A pairs based on review analysis
    """
    print("[PROCESS] Generating Q&A pairs...")
    
    qa_pairs = []
    
    # Analyze common terms and topics
    all_text = " ".join(df["Processed_Review"].tolist())
    topics = extract_key_topics(df, phase2_data)
    
    # Q1: Battery Life
    battery_keywords = ["battery", "charging", "charge", "power", "hours", "backup"]
    battery_reviews = df[df["Processed_Review"].str.contains("|".join(battery_keywords), case=False, na=False)]
    
    if len(battery_reviews) > 0:
        battery_mentions = len(battery_reviews)
        positive_battery = len(battery_reviews[battery_reviews["Processed_Review"].str.contains("good|great|excellent|long|best|awesome", case=False, na=False)])
        negative_battery = len(battery_reviews[battery_reviews["Processed_Review"].str.contains("bad|poor|short|worst|issue|problem", case=False, na=False)])
        
        battery_sample = battery_reviews["Review"].iloc[0] if len(battery_reviews) > 0 else ""
        
        answer = f"Battery performance is mentioned in {battery_mentions} reviews ({(battery_mentions/len(df)*100):.1f}% of total). "
        if positive_battery > negative_battery:
            answer += f"Majority sentiment is POSITIVE ({positive_battery} positive vs {negative_battery} negative mentions). "
            answer += "Customers generally appreciate the battery backup and charging speed."
        elif negative_battery > positive_battery:
            answer += f"Sentiment is NEGATIVE ({negative_battery} negative vs {positive_battery} positive mentions). "
            answer += "Some customers report issues with battery performance."
        else:
            answer += "Sentiment is MIXED with both positive and negative feedback."
        
        qa_pairs.append({
            "question": "How is the battery life and charging performance?",
            "answer": answer,
            "supporting_data": {
                "total_mentions": battery_mentions,
                "positive_count": positive_battery,
                "negative_count": negative_battery,
                "sample_review": battery_sample[:150]
            }
        })
    
    # Q2: Sound Quality
    sound_keywords = ["sound", "audio", "bass", "music", "quality", "volume"]
    sound_reviews = df[df["Processed_Review"].str.contains("|".join(sound_keywords), case=False, na=False)]
    
    if len(sound_reviews) > 0:
        sound_mentions = len(sound_reviews)
        positive_sound = len(sound_reviews[sound_reviews["Processed_Review"].str.contains("good|great|excellent|amazing|best|clear|awesome", case=False, na=False)])
        negative_sound = len(sound_reviews[sound_reviews["Processed_Review"].str.contains("bad|poor|worst|issue|problem|low", case=False, na=False)])
        
        sound_sample = sound_reviews["Review"].iloc[0] if len(sound_reviews) > 0 else ""
        
        answer = f"Sound quality is discussed in {sound_mentions} reviews ({(sound_mentions/len(df)*100):.1f}% of total). "
        if positive_sound > negative_sound:
            answer += f"POSITIVE sentiment dominates ({positive_sound} positive vs {negative_sound} negative). "
            answer += "Customers are satisfied with audio quality, bass, and volume levels."
        else:
            answer += f"Mixed to negative sentiment ({negative_sound} complaints vs {positive_sound} positive). "
            answer += "Some users have concerns about audio performance."
        
        qa_pairs.append({
            "question": "Is the sound quality good?",
            "answer": answer,
            "supporting_data": {
                "total_mentions": sound_mentions,
                "positive_count": positive_sound,
                "negative_count": negative_sound,
                "sample_review": sound_sample[:150]
            }
        })
    
    # Q3: Build Quality & Durability
    build_keywords = ["build", "quality", "durable", "material", "sturdy", "plastic", "strong"]
    build_reviews = df[df["Processed_Review"].str.contains("|".join(build_keywords), case=False, na=False)]
    
    if len(build_reviews) > 0:
        build_mentions = len(build_reviews)
        positive_build = len(build_reviews[build_reviews["Processed_Review"].str.contains("good|great|excellent|solid|premium|best", case=False, na=False)])
        negative_build = len(build_reviews[build_reviews["Processed_Review"].str.contains("bad|poor|cheap|fragile|break|broke", case=False, na=False)])
        
        build_sample = build_reviews["Review"].iloc[0] if len(build_reviews) > 0 else ""
        
        answer = f"Build quality is mentioned in {build_mentions} reviews ({(build_mentions/len(df)*100):.1f}% of total). "
        if positive_build > negative_build:
            answer += f"POSITIVE feedback ({positive_build} positive vs {negative_build} negative). "
            answer += "Users appreciate the solid construction and material quality."
        else:
            answer += f"Concerns exist ({negative_build} negative vs {positive_build} positive). "
            answer += "Some customers question the durability and build materials."
        
        qa_pairs.append({
            "question": "Is the product durable and well-built?",
            "answer": answer,
            "supporting_data": {
                "total_mentions": build_mentions,
                "positive_count": positive_build,
                "negative_count": negative_build,
                "sample_review": build_sample[:150]
            }
        })
    
    # Q4: Value for Money
    value_keywords = ["price", "worth", "value", "money", "affordable", "expensive", "cheap"]
    value_reviews = df[df["Processed_Review"].str.contains("|".join(value_keywords), case=False, na=False)]
    
    if len(value_reviews) > 0:
        value_mentions = len(value_reviews)
        positive_value = len(value_reviews[value_reviews["Processed_Review"].str.contains("good|great|worth|affordable|reasonable|best", case=False, na=False)])
        negative_value = len(value_reviews[value_reviews["Processed_Review"].str.contains("expensive|overpriced|not worth|waste", case=False, na=False)])
        
        value_sample = value_reviews["Review"].iloc[0] if len(value_reviews) > 0 else ""
        
        answer = f"Value for money is discussed in {value_mentions} reviews ({(value_mentions/len(df)*100):.1f}% of total). "
        if positive_value > negative_value:
            answer += f"POSITIVE sentiment ({positive_value} positive vs {negative_value} negative). "
            answer += "Customers feel the product offers good value and is worth the price."
        else:
            answer += f"Mixed sentiment ({negative_value} concerns vs {positive_value} positive). "
            answer += "Some users feel it's overpriced for what it offers."
        
        qa_pairs.append({
            "question": "Is it worth the price? Good value for money?",
            "answer": answer,
            "supporting_data": {
                "total_mentions": value_mentions,
                "positive_count": positive_value,
                "negative_count": negative_value,
                "sample_review": value_sample[:150]
            }
        })
    
    # Q5: Overall Recommendation
    recommend_keywords = ["recommend", "buy", "purchase", "suggest"]
    positive_overall = len(df[df["Processed_Review"].str.contains("good|great|excellent|amazing|best|awesome|love|perfect", case=False, na=False)])
    negative_overall = len(df[df["Processed_Review"].str.contains("bad|poor|worst|hate|terrible|disappoint|waste", case=False, na=False)])
    
    answer = f"Based on analysis of {len(df)} reviews: "
    satisfaction_rate = (positive_overall / len(df)) * 100
    
    if satisfaction_rate > 60:
        answer += f"HIGHLY RECOMMENDED. {satisfaction_rate:.1f}% of reviews express positive sentiment. "
        answer += f"Common praise: {', '.join(topics.get('important_terms', ['quality', 'performance'])[:3])}."
    elif satisfaction_rate > 40:
        answer += f"MODERATELY RECOMMENDED. {satisfaction_rate:.1f}% positive sentiment. Mixed feedback suggests product works well for some but has issues."
    else:
        answer += f"NOT RECOMMENDED. Only {satisfaction_rate:.1f}% positive sentiment. Significant concerns raised by customers."
    
    # Add top representative review as evidence
    if representative_reviews:
        answer += f" Most common feedback theme: '{representative_reviews[0]['review_original'][:100]}...'"
    
    qa_pairs.append({
        "question": "Would you recommend this product? Overall verdict?",
        "answer": answer,
        "supporting_data": {
            "total_reviews": len(df),
            "positive_reviews": positive_overall,
            "negative_reviews": negative_overall,
            "satisfaction_rate": f"{satisfaction_rate:.1f}%",
            "top_feedback": representative_reviews[0]['review_original'][:200] if representative_reviews else ""
        }
    })
    
    print(f"[SUCCESS] Generated {len(qa_pairs)} Q&A pairs")
    
    return qa_pairs

def run_phase3_analysis():
    """Main Phase 3 execution"""
    print("[INFO] ========== PHASE 3: ADVANCED ANALYSIS ==========")
    
    # Load data
    df, phase2_data = load_data()
    print(f"[OK] Loaded {len(df)} reviews and Phase 2 results")
    
    # Task 1: Review Summarization using Similarity Index
    representative_reviews, similarity_matrix, df_clustered = extract_representative_reviews(df, n_clusters=5, top_reviews=5)
    
    # Calculate overall similarity statistics
    avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
    max_similarity = np.max(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
    
    # Task 2: Question Answering
    qa_pairs = generate_qa_pairs(df_clustered, phase2_data, representative_reviews)
    
    # Compile results
    results = {
        "review_summarization": {
            "method": "TF-IDF vectorization + K-Means clustering + Cosine similarity",
            "num_clusters": len(set(df_clustered["cluster"])),
            "total_reviews_analyzed": len(df),
            "avg_similarity_score": float(avg_similarity),
            "max_similarity_score": float(max_similarity),
            "representative_reviews": representative_reviews,
            "cluster_distribution": df_clustered["cluster"].value_counts().to_dict()
        },
        "question_answering": {
            "method": "Keyword-based extraction + Sentiment analysis + Data synthesis",
            "num_questions": len(qa_pairs),
            "qa_pairs": qa_pairs
        },
        "summary": {
            "total_reviews": len(df),
            "unique_clusters": len(set(df_clustered["cluster"])),
            "key_findings": [
                f"Identified {len(representative_reviews)} representative review themes",
                f"Average review similarity: {avg_similarity:.3f}",
                f"Generated {len(qa_pairs)} data-driven Q&A pairs",
                f"Most common theme has {representative_reviews[0]['cluster_size']} similar reviews" if representative_reviews else ""
            ]
        }
    }
    
    # Save results
    output_file = "results/phase3_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[SUCCESS] Phase 3 analysis complete! Results saved to {output_file}")
    
    return results

if __name__ == "__main__":
    results = run_phase3_analysis()
    print(json.dumps(results, indent=2, ensure_ascii=False))
