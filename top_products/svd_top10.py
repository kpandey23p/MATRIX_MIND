import pandas as pd
import gzip
import json
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from urllib.request import urlretrieve
import pickle
import pandas as pd


review_path = "Appliances.jsonl.gz"
meta_path = "meta_Appliances.jsonl.gz"

urlretrieve(review_url, review_path)
urlretrieve(meta_url, meta_path)

# === Load review dataset ===
def load_reviews(path, max_rows=None):
    reviews = []
    with gzip.open(path, 'rt') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            reviews.append((data['user_id'], data['parent_asin'], data['rating']))
            if max_rows and i >= max_rows:
                break
    return pd.DataFrame(reviews, columns=["user_id", "asin", "rating"])

# Load a limited number of reviews for memory efficiency
df_reviews = load_reviews(review_path, max_rows=500000)

# === Load metadata and build ASIN → Title mapping ===
def load_metadata(path):
    asin_title = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            data = json.loads(line)
            asin = data.get("parent_asin") or data.get("asin")
            title = data.get("title", "Unknown Title")
            if asin:
                asin_title[asin] = title
    return asin_title

asin_to_title = load_metadata(meta_path)

# === Create pivot table and encode users/items ===
user_ids = df_reviews['user_id'].astype("category").cat.codes
item_ids = df_reviews['asin'].astype("category").cat.codes

ratings_matrix = csr_matrix((df_reviews['rating'], (user_ids, item_ids)))

# === Apply Truncated SVD ===
svd = TruncatedSVD(n_components=50, random_state=42)
user_features = svd.fit_transform(ratings_matrix)
item_features = svd.components_.T

user_map = dict(enumerate(df_reviews['user_id'].astype("category").cat.categories))
item_map = dict(enumerate(df_reviews['asin'].astype("category").cat.categories))
reverse_user_map = {v: k for k, v in user_map.items()}

# === Recommend and show titles ===
def recommend_for_user(user_id, top_n=10):
    if user_id not in reverse_user_map:
        return []
    user_idx = reverse_user_map[user_id]
    scores = item_features @ user_features[user_idx]
    top_items = np.argsort(-scores)[:top_n]
    asins = [item_map[i] for i in top_items]
    return [(asin, asin_to_title.get(asin, "Title not found")) for asin in asins]

# === Example Usage ===
example_user = df_reviews['user_id'].iloc[0]
recommendations = recommend_for_user(example_user)

print(f"Top recommendations for user {example_user}:\n")
for asin, title in recommendations:
    print(f"- {asin}: {title}")



# Save model and metadata
with open("svd_model.pkl", "wb") as f:
        pickle.dump(svd, f)

with open("user_map.pkl", "wb") as f:
    pickle.dump(user_map, f)

with open("item_map.pkl", "wb") as f:
    pickle.dump(item_map, f)

# meta_df must be in dict format {asin: {"title": ...}}
pd.to_pickle(meta_df, "meta_df.pkl")

print("✅ All files saved successfully!")




