from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import numpy as np
import pandas as pd
import pickle

# Initialize FastAPI app
app = FastAPI()

# Allow CORS (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model components
try:
    with open("user_map.pkl", "rb") as f:
        user_map = pickle.load(f)

    with open("item_map.pkl", "rb") as f:
        item_map = pickle.load(f)

    item_map_inv = {v: k for k, v in item_map.items()}

    with open("user_features.pkl", "rb") as f:
        user_features = pickle.load(f)

    with open("item_features.pkl", "rb") as f:
        item_features = pickle.load(f)

    asin_to_title = pd.read_pickle("meta_df.pkl")
except Exception as e:
    print("❌ Model or metadata loading failed:", e)
    user_map = item_map = item_map_inv = user_features = item_features = asin_to_title = None

@app.get("/recommend", response_model=List[dict])
def recommend(user_id: str):
    if user_map is None or user_features is None or item_features is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    if user_id not in user_map:
        raise HTTPException(status_code=404, detail="User not found.")

    uid = user_map[user_id]
    user_vec = user_features[uid]

    predictions = [
        (iid, np.dot(user_vec, item_features[iid]))
        for iid in range(item_features.shape[0])
    ]

    top_k = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]

    recommendations = []
    for iid, score in top_k:
        asin = item_map_inv[iid]
        title = asin_to_title.get(asin, "Unknown Product")
        recommendations.append({"asin": asin, "title": title})


    return recommendations