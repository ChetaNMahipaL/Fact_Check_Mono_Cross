import ast
import os
import re
import string
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sentence_transformers import SentenceTransformer
import json
import torch

# Set the environment variable to disable unnecessary warnings related to tokenizers
os.environ['TSAN_OPTIONS'] = 'ignore_noninstrumented_modules=1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load SentenceTransformer model
model_name = 'sentence-transformers/sentence-t5-large'
model = SentenceTransformer(model_name)
# Check if CUDA (GPU) is available and move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

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

# Load fact checks dataset
our_dataset_path = '.'
fact_checks_path = os.path.join(our_dataset_path, 'fact_checks.csv')
df_fact_checks = pd.read_csv(fact_checks_path).fillna('').set_index('fact_check_id')

# Parse and clean data
parse_col = lambda s: ast.literal_eval(s.replace('\n', '\\n')) if s else s
for col in ['claim', 'title']:
    df_fact_checks[col] = df_fact_checks[col].apply(parse_col)

df_fact_checks[['claim', 'translated_claim', 'language']] = pd.DataFrame(df_fact_checks['claim'].tolist(), index=df_fact_checks.index)
df_fact_checks['claim'] = df_fact_checks['claim'].apply(remove_emojis_and_punctuation)
df_fact_checks['translated_claim'] = df_fact_checks['translated_claim'].apply(remove_emojis_and_punctuation)

# Extract claims
claims_list = df_fact_checks['translated_claim'][:30000].tolist()

# Function to generate embeddings using the available device (GPU/CPU)
def generate_embeddings_with_device(claims_list, model, device):
    # Move data to device (GPU if available, otherwise CPU)
    claims_list = [claim for claim in claims_list]  # Convert to a list of strings
    embeddings = model.encode(claims_list, show_progress_bar=True, device=device)
    return embeddings

# Generate embeddings using SentenceTransformer (on GPU if available)
embeddings = generate_embeddings_with_device(claims_list, model, device)

# Normalize the embeddings
embeddings = embeddings.astype(np.float32)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Create FAISS index for embeddings
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)  # Add the embeddings to the index

# Load filtered posts and grouped pairs
filtered_posts_path = os.path.join(our_dataset_path, 'filtered_posts_translated.json')
grouped_pairs_path = os.path.join(our_dataset_path, 'grouped_pairs.json')

with open(filtered_posts_path, 'r') as f:
    filtered_posts = json.load(f)

with open(grouped_pairs_path, 'r') as f:
    grouped_pairs = json.load(f)

# Prepare posts and ground truth facts
posts = [post['translated_ocr'] for post in filtered_posts]
post_ids = [post['idx'] for post in filtered_posts]
ground_truth = [grouped_pairs.get(str(post_id), []) for post_id in post_ids]

# Function to find top X closest claims
def find_top_x_closest_claims(input_claim, index, model, x=5):
    # Preprocess input claim (remove emojis, punctuation, etc.)
    input_claim_processed = remove_emojis_and_punctuation(input_claim)
    
    # Generate Sentence-BERT embedding for the input claim
    input_vec = model.encode([input_claim_processed]).astype(np.float32)  # Ensure it's float32
    input_vec = input_vec.reshape(1, -1)
    
    # Search the FAISS index for the top-x closest claims
    _, indices = index.search(input_vec, x)
    
    # Get the closest claims from the DataFrame using the indices
    closest_claims = df_fact_checks.iloc[indices[0]].reset_index()
    
    return closest_claims

# Function to evaluate predictions
def evaluate_predictions(predictions, ground_truth, top_x):
    y_true = []
    y_pred = []
    true = 0
    false = 0
    
    for truth, preds in zip(ground_truth, predictions):
        truth_set = set(truth)
        preds_set = set(preds[:top_x])
        flag = 0
        for a in preds_set:
            if a in truth_set and flag == 0:
                true += 1
                flag = 1
        if flag == 0:
            false += 1
        
        y_true += [1 if fact in truth_set else 0 for fact in preds]
        y_pred += [1] * len(preds)
    
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')

    print("Custom Metric (True/False Ratio):", true / (true + false))
    
    return f1, accuracy, recall

# Find closest claims for each post
predictions = []

# Example of usage
for post in posts:
    closest_claims = find_top_x_closest_claims(post, index, model, x=10)
    predictions.append(closest_claims['fact_check_id'].tolist())

# Evaluate metrics
f1, accuracy, recall = evaluate_predictions(predictions, ground_truth, top_x=10)

# Print results
print(f"Top-5 Evaluation Metrics:")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
