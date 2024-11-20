import os
import json
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import re
import pandas as pd
import string

# Function to remove emojis and punctuation from text
def remove_emojis_and_punctuation(text):
    if not isinstance(text, str):
        return text
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('\t', ' ').replace('\n', ' ')
    return text

# Function to find the top X closest claims
def find_top_x_closest_claims(input_claim, vectorizer, index, top_x=5):
    input_claim_processed = remove_emojis_and_punctuation(input_claim)
    input_vec = vectorizer.transform([input_claim_processed]).toarray().astype(np.float32)
    _, indices = index.search(input_vec, top_x)
    closest_claims = df_fact_checks.iloc[indices[0]].reset_index()
    return closest_claims

# Function to evaluate predictions
def evaluate_predictions(predictions, ground_truth, top_x):
    y_true = []
    y_pred = []
    
    for truth, preds in zip(ground_truth, predictions):
        truth_set = set(truth)
        preds_set = set(preds[:top_x])
        
        y_true += [1 if fact in truth_set else 0 for fact in preds]
        y_pred += [1] * len(preds)
    
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    
    return f1, accuracy, recall

# Function to process a single post in parallel
def process_post(post, top_x, claims_list, vectorizer, dimension):
    # Load FAISS index and vectorizer in each worker
    index = faiss.IndexFlatL2(dimension)
    tfidf_matrix = vectorizer.transform(claims_list).toarray()
    index.add(np.array(tfidf_matrix, dtype=np.float32))
    
    try:
        closest_claims = find_top_x_closest_claims(post, vectorizer, index, top_x)
        return closest_claims['fact_check_id'].tolist()
    except Exception as e:
        print(f"Error processing post: {e}")
        return []

def parallel_find_predictions(posts, claims_list, vectorizer, top_x, dimension, n_jobs=8):
    """
    Finds the top X closest claims for all posts in parallel.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_post)(post, top_x, claims_list, vectorizer, dimension) for post in tqdm(posts, desc="Processing posts")
    )
    return results

# Load data
our_dataset_path = '.'

filtered_posts_path = os.path.join(our_dataset_path, 'filtered_posts.json')
grouped_pairs_path = os.path.join(our_dataset_path, 'grouped_pairs.json')
fact_checks_path = os.path.join(our_dataset_path, 'fact_checks.csv')

with open(filtered_posts_path, 'r') as f:
    filtered_posts = json.load(f)

with open(grouped_pairs_path, 'r') as f:
    grouped_pairs = json.load(f)

df_fact_checks = pd.read_csv(fact_checks_path).fillna('').set_index('fact_check_id')

# Prepare posts and ground truth facts
posts = [post['ocr'] for post in filtered_posts]
post_ids = [post['idx'] for post in filtered_posts]
ground_truth = [grouped_pairs.get(str(post_id), []) for post_id in post_ids]

# Prepare the vectorizer
claims_list = df_fact_checks['claim'][:30000].tolist()
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(claims_list)

# FAISS Index: Using TF-IDF embeddings
dimension = len(vectorizer.get_feature_names_out())

# Parallelize finding predictions
top_x = 2
n_jobs = 16
predictions = parallel_find_predictions(posts, claims_list, vectorizer, top_x, dimension, n_jobs=n_jobs)

# Evaluate metrics
f1, accuracy, recall = evaluate_predictions(predictions, ground_truth, top_x)

# Display results
print(f"Top-{top_x} Evaluation Metrics:")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
