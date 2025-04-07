# 🎯 Product Recommendation using Truncated SVD

This project contains **two separate implementations** of recommendation systems based on **Truncated SVD**:

1. **Model A: Top Products** – A FastAPI-based backend with a simple frontend UI that returns top-N recommendations.
2. **Model B: Masked Matrix** – An evaluation script that masks known ratings and tests how well the model predicts them.

---


---

## 🧠 Model A: Top Products

This model builds a recommendation system that returns the **top 10 product recommendations** for a given user via a fast API.

### ⚙️ How It Works

- Uses Truncated SVD on the user-item rating matrix.
- Generates top-N items for a user based on latent factor scores.
- ASINs are mapped to product titles using metadata.
