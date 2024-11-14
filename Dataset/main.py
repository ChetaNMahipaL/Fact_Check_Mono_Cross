# %%
import ast
import os
import re
import uuid

import pandas as pd
import string
import weaviate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from weaviate.embedded import EmbeddedOptions
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer

# from sentence_transformers import SentenceTransformer


def remove_emojis_and_punctuation(text):
    # Remove emojis

    if not isinstance(text, str):
        print(text[2])
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
    
    # Remove emojis
    text = emoji_pattern.sub(r'', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('\t', ' ').replace('\n', ' ')

    
    return text

# %%
"""
This is just a quick script that is able to load the files. Just using pandas can be tricky because of the newline characters in the text data. Here it is handled via the `parse_col` method.
"""

import ast
import os

import pandas as pd

our_dataset_path = '.'

posts_path = os.path.join(our_dataset_path, 'posts.csv')
fact_checks_path = os.path.join(our_dataset_path, 'fact_checks.csv')

parse_col = lambda s: ast.literal_eval(s.replace('\n', '\\n')) if s else s

df_fact_checks = pd.read_csv(fact_checks_path).fillna('').set_index('fact_check_id')
for col in ['claim', 'title']:
    df_fact_checks[col] = df_fact_checks[col].apply(parse_col)


df_fact_checks[['claim', 'translated_claim', 'language']] = pd.DataFrame(df_fact_checks['claim'].tolist(), index=df_fact_checks.index)
df_fact_checks['claim'] = df_fact_checks['claim'].apply(remove_emojis_and_punctuation)
df_fact_checks['translated_claim'] = df_fact_checks['translated_claim'].apply(remove_emojis_and_punctuation)

df_fact_check_claims = df_fact_checks[['claim', 'translated_claim', 'language']]


# if 'fact_check_id' not in df_fact_check_claims.columns:
#     df_fact_check_claims = df_fact_check_claims.reset_index()

# Save the extracted data to a new CSV file
output_path = os.path.join(our_dataset_path, 'fact_check_claims.csv')
df_fact_check_claims.to_csv(output_path, index_label='fact_check_id')

print(f"Extracted fact check claims saved to {output_path}")

# %%
limited_df = df_fact_checks.head(50)
output_path = os.path.join(our_dataset_path, 'sample_fact_check_claims.csv')

limited_df.to_csv(output_path, index_label='fact_check_id')


# %%


# %%


client = weaviate.Client(
    embedded_options=EmbeddedOptions()
)


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_fact_checks['claim'].tolist())

# Create documents for vector st
for i, row in limited_df.iterrows():
    embedding = tfidf_matrix[i].toarray().flatten().tolist()  # Convert sparse matrix row to list
    new_uuid = str(uuid.uuid4()) 
    original_index = row.name 
    print(row['claim'])
    
    # Create document and insert into Weaviate
    client.data_object.create(
        data_object={
            "claim": row['claim'],  # Cleaned claim text
            "translated_claim": row['translated_claim'],  # Include the translated claim
            "language": row['language'][0][0],  # Include the language
            "embedding": embedding,  # BoW embedding
            "original_index": original_index  # Store the original index for later reference
        },
        class_name="YourClassName",  # Replace with your desired class name
        uuid=new_uuid  # Use the new valid UUID
    )

print("Vector database has been successfully populated.")


# %%
from langchain.vectorstores import Weaviate as LangchainWeaviate
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# %%
vectorstore = LangchainWeaviate(
    client=client,
    index_name="YourClassName",  # Use the class name you specified in Weaviate
    text_key="claim"  # The key that contains the text to be embedded and retrieved
)
retriever = vectorstore.as_retriever()


# %%
# Input sentence
input_sentence = "150 million people"

# Transform the input sentence into a vector
input_vector = vectorizer.transform([input_sentence]).toarray().flatten().tolist()


nearest_neighbors = (
    client.query.get("YourClassName", ["claim", "translated_claim", "language", "original_index"])
    .with_near_vector({"vector": input_vector})
    .with_limit(1)  # Get the closest single match
    .do()
)
# Step 3: Extract the closest index and information
if nearest_neighbors['data']['Get']['YourClassName']:
    closest_match = nearest_neighbors['data']['Get']['YourClassName'][0]
    print(f"Closest match found: {closest_match}")
else:
    print("No match found, but you can still retrieve the closest claim.")

# %%
all_claims = client.query.get("YourClassName", ["claim", "translated_claim", "language", "original_index"]).do()
with open("./db.txt" ,"w") as f:
    for claim_record in all_claims['data']['Get']['YourClassName']:
        claim_text = claim_record['claim']
        f.write(claim_text+"\n\n")

# %%
input_sentence = "COV1D19 death"

def calculate_word_similarity(input_sentence, claim):
    input_words = set(input_sentence.lower().split())
    claim_words = set(claim.lower().split())
    # Count the number of shared words
    return len(input_words.intersection(claim_words))

# Fetch all claims from Weaviate
all_claims = client.query.get("YourClassName", ["claim", "translated_claim", "language", "original_index"]).do()

# Initialize variables to track the best match
best_score = 0
best_claim = None

# Loop through all claims to find the most similar one
for claim_record in all_claims['data']['Get']['YourClassName']:
    claim_text = claim_record['claim']
    # print(claim_text)
    score = calculate_word_similarity(input_sentence, claim_text)
    
    if score > best_score:
        best_score = score
        best_claim = claim_record

# Output the best matching claim
if best_claim:
    print(f"Best matching claim found: {best_claim}")
else:
    print("No matching claim found.")





