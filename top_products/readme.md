title: SVD Recommender System

sections:
  -  🚀 How to Run Model A (SVD Recommender API)
    steps:
      -  1️⃣ Train the Model
        description: |
          Run the following script to train the SVD model and generate required files:
        code: |
          python svd_top10.py
        outputs:
          - model.pkl – Trained TruncatedSVD model
          - user_map.pkl – Mapping of user IDs to matrix indices
          - item_map.pkl – Mapping of ASINs to matrix indices
          - meta_df.pkl – ASIN to product title metadata

      -  2️⃣ Start the API Server
        description: |
          Launch the FastAPI backend with:
        code: |
          uvicorn svd_back:app --reload
        note: Server will run at http://127.0.0.1:8000

      - title: 3️⃣ Access the Frontend
        description: |
          Open the `index.html` file in your browser:
          - Enter a valid user ID (from the dataset)
          - Click "Get Recommendations"
          - See the top 10 recommended products with titles

