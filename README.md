# Image Search for E-commerce products

## CLIP-Based Image Search (Fine-tuned with Custom Descriptions)

This project implements an **image search engine using a fine-tuned [CLIP](https://openai.com/research/clip)** model. It allows users to enter a **text query** and retrieves the **most relevant images** using cosine similarity between query and image embeddings.

---

## Features

- ✅ Fine-tuned CLIP model on custom dataset (image + product display names)
- ✅ Retrieves top-k images based on cosine similarity with query
- ✅ Fast similarity search using precomputed image embeddings
- ✅ Interactive **Streamlit** UI for seamless querying and visualization
- ✅ Optional baseline using **BM25** text search on product descriptions

---

## How It Works

1. **Data**: Each image is paired with a human-readable display name (e.g., *"Red Leather Handbag"*).
2. **Fine-Tuning**: CLIP is fine-tuned on positive and negative (image, text) pairs using contrastive loss.
3. **Embedding Index**: All image embeddings are precomputed and stored.
4. **Retrieval**: Given a query (e.g., *"bracelets"*), it is encoded, and top images are ranked via cosine similarity.
5. **UI**: Results are shown in a Streamlit web app with image grid and score annotations.

---
Sample search using CLIP embeddings:
<img width="3600" height="1200" alt="results_Sport shoes" src="https://github.com/user-attachments/assets/06343260-c056-4700-9009-0982611ab0d5" />

<img width="3600" height="1200" alt="results_Men Jeans" src="https://github.com/user-attachments/assets/5f24118a-cc6b-4c0a-8167-04511eb4b672" />
