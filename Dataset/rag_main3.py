import ast
import os
import re
import string
import pandas as pd
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score, recall_score
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

from joblib import Parallel, delayed

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# Function to remove emojis and punctuation from text
def remove_emojis_and_punctuation(text):
    if not isinstance(text, str):
        # print(text[2])
        exit(1)
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


# Read in the CSV data for fact checks
our_dataset_path = '.'
fact_checks_path = os.path.join(our_dataset_path, 'fact_checks.csv')
df_fact_checks = pd.read_csv(fact_checks_path).fillna('').set_index('fact_check_id')

# Parse and clean data
parse_col = lambda s: ast.literal_eval(s.replace('\n', '\\n')) if s else s
for col in ['claim', 'title']:
    df_fact_checks[col] = df_fact_checks[col].apply(parse_col)
# print(df_fact_checks['claim'])
# print(df_fact_checks['claim'][0])

df_fact_checks[['claim', 'translated_claim', 'language']] = pd.DataFrame(df_fact_checks['claim'].tolist(), index=df_fact_checks.index)
df_fact_checks['claim'] = df_fact_checks['claim'].apply(remove_emojis_and_punctuation)
df_fact_checks['translated_claim'] = df_fact_checks['translated_claim'].apply(remove_emojis_and_punctuation)

# Extract the claims for use in the search
df_fact_check_claims = df_fact_checks[['claim', 'translated_claim', 'language']]

# Generate embeddings for the claims using OpenAI embeddings

def process_claims_batch(claims_batch, vectorizer):
    try:
        # Transform the batch into a dense matrix and return it
        return vectorizer.transform(claims_batch).toarray()
    except Exception as e:
        print(f"Error processing batch: {e}")
        return None  # Return None in case of error, this helps to catch failed batches


def populate_faiss_index_parallel(claims_list, vectorizer, index, n_jobs=16):
    # Split the claims into batches for parallel processing
    batch_size = len(claims_list) // n_jobs
    batches = [claims_list[i:i + batch_size] for i in range(0, len(claims_list), batch_size)]
    
    # Parallelize the batch processing using joblib
    results = Parallel(n_jobs=n_jobs)(delayed(process_claims_batch)(batch, vectorizer) for batch in batches)
    
    # Check if any batch processing returned None or empty results
    if any(result is None for result in results):
        print("One or more batches failed to process.")
        return None  # Stop if any batch failed
    
    # Flatten the results and convert to the correct format for FAISS
    dense_matrix = np.vstack(results).astype(np.float32)
    
    # Check if the dense_matrix is empty or malformed
    if dense_matrix.size == 0:
        print("Dense matrix is empty after vstack.")
        return None
    
    # Add the vectors to the FAISS index
    index.add(dense_matrix)

claims_list = df_fact_check_claims['claim'][:30000].tolist()
vectorizer = TfidfVectorizer(stop_words='english')  #issue of word covid
tfidf_matrix = vectorizer.fit_transform(claims_list)
dense_matrix = tfidf_matrix.toarray()
# dense_matrix = model.encode(claims_list)

dimension = dense_matrix.shape[1] 
index = faiss.IndexFlatL2(dimension)
# index.add(np.array(dense_matrix, dtype=np.float32))
populate_faiss_index_parallel(claims_list, vectorizer, index, n_jobs=16)
# Create embeddings for the claims



import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score
from tqdm import tqdm

# Load filtered posts and grouped pairs from JSON
filtered_posts_path = os.path.join(our_dataset_path, 'filtered_posts.json')
grouped_pairs_path = os.path.join(our_dataset_path, 'grouped_pairs.json')

with open(filtered_posts_path, 'r') as f:
    filtered_posts = json.load(f)

with open(grouped_pairs_path, 'r') as f:
    grouped_pairs = json.load(f)

# Prepare posts and ground truth facts
posts = [post['ocr'] for post in filtered_posts]
post_ids = [post['idx'] for post in filtered_posts]
ground_truth = [grouped_pairs.get(str(post_id), []) for post_id in post_ids]

# Function to find top X closest claims
def find_top_x_closest_claims(input_claim, index, vectorizer, x=5):
    input_claim_processed = remove_emojis_and_punctuation(input_claim)
    # print(input_claim)
    input_vec = vectorizer.transform([input_claim_processed]).toarray().astype(np.float32)
    _, indices = index.search(input_vec, x)
    closest_claims = df_fact_checks.iloc[indices[0]].reset_index()
    # print(closest_claims)
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
        # print(truth_set)
        # print(preds_set)
        flag=0
        for  a in preds_set:
            if a in truth_set and flag==0:
                true+=1
                flag=1
        if flag==0:
            false+=1

        


        
        # Generate labels (1 for true facts, 0 otherwise)
        y_true += [1 if fact in truth_set else 0 for fact in preds]
        y_pred += [1] * len(preds)
    
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')

    print("my:", true/(true+false))
    
    return f1, accuracy, recall

# Evaluate for top X closest claims

for top_x in [5]:
    
    predictions = []


    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_post(post):
        # Function to process each post and get closest claims
        return find_top_x_closest_claims(post, index, vectorizer, x=top_x)['fact_check_id'].tolist()


    predictions = []
    with ThreadPoolExecutor(max_workers=24) as executor:
        # Create a dictionary to map future to post
        futures = {executor.submit(process_post, post): post for post in posts}
        
        # Iterate over the results as they complete
        for future in tqdm(as_completed(futures), total=len(posts), desc="Finding closest claims"):
            result = future.result()  # Get the result from the future
            predictions.append(result)

    # Evaluate metrics
    f1, accuracy, recall = evaluate_predictions(predictions, ground_truth, top_x)

    print(f"Top-{top_x} Evaluation Metrics:")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")