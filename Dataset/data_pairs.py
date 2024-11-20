import os
import pandas as pd
import ast
import json

# Function to clean text
our_dataset_path='.'

def remove_emojis_and_punctuation(text):
    import re
    # Remove emojis
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load pairs data
pair_path = os.path.join('.', 'pairs.csv')
df_pair = pd.read_csv(pair_path)

# # Filter for fact IDs < 30,000
filtered_pairs = df_pair[df_pair['fact_check_id'] < 30000]

# # Group facts by post_id
post_to_facts = filtered_pairs.groupby('post_id')['fact_check_id'].apply(list).to_dict()

# # Save grouped pairs to a JSON file
# grouped_pairs_path = os.path.join('.', 'grouped_pairs.json')
# with open(grouped_pairs_path, 'w') as f:
#     json.dump(post_to_facts, f, indent=4)
# print(f"Grouped pairs saved to {grouped_pairs_path}")

# Load posts data
post_path = os.path.join(our_dataset_path, 'posts.csv')
df_posts = pd.read_csv(post_path).fillna('').set_index('post_id')

# Parse and clean posts data
parse_col = lambda s: ast.literal_eval(s.replace('\n', '\\n')) if s else s
for col in ['ocr', 'text']:
    df_posts[col] = df_posts[col].apply(parse_col)
print(df_posts['ocr'][0][0])
print(len(df_posts['ocr'][0][0]))

def safe_extract_first_element(ocr_entry):
    try:
        return ocr_entry[0]
    except (TypeError, IndexError):  # Handle cases where ocr_entry is invalid
        return ("","","")

# Apply the helper function
df_posts['ocr'] = df_posts['ocr'].apply(safe_extract_first_element)



df_posts[['ocr', 'translated_ocr', 'language']] = pd.DataFrame(df_posts['ocr'].tolist(), index=df_posts.index)
df_posts['ocr'] = df_posts['ocr'].apply(remove_emojis_and_punctuation)
df_posts['idx'] = df_posts.index
df_posts['translated_ocr'] = df_posts['translated_ocr'].apply(remove_emojis_and_punctuation)

# Filter posts for fact IDs < 30,000
filtered_posts = [
    {"idx": post_id, "translated_ocr": df_posts.loc[post_id, 'translated_ocr']}
    for post_id in post_to_facts.keys()
    if post_id in df_posts.index
]

# Save filtered posts to a JSON file
filtered_posts_path = os.path.join(our_dataset_path, 'filtered_posts_translated.json')
with open(filtered_posts_path, 'w') as f:
    json.dump(filtered_posts, f, indent=4)
print(f"Filtered posts saved to {filtered_posts_path}")

# Save grouped pairs to a JSON file
# grouped_pairs_path = os.path.join(our_dataset_path, 'grouped_pairs.json')
# with open(grouped_pairs_path, 'w') as f:
#     json.dump(post_to_facts, f, indent=4)
# print(f"Grouped pairs saved to {grouped_pairs_path}")