# Image Search for E-commerce products

## CLIP-Based Image Search (Fine-tuned with Custom Descriptions)

This project implements an **image search engine using a fine-tuned [CLIP](https://openai.com/research/clip)** model. It allows users to **upload an image** or enter a **text query** and retrieves the **most relevant images** using cosine similarity between query and image embeddings.

---

## Features

- ✅ Fine-tuned CLIP model on custom dataset (image + product display names)
- ✅ Retrieves top-k images based on cosine similarity with query/Image
- ✅ Fast similarity search using precomputed image embeddings
- ✅ Interactive **Streamlit** UI for seamless querying and visualization
- ✅ Optional baseline using **BM25** text search on product descriptions

---

## How It Works

1. **Data**: Each image is paired with a human-readable display name (e.g., *"Red Leather Handbag"*).
2. **Fine-Tuning**: CLIP is fine-tuned on positive and negative (image, text) pairs using contrastive loss.
3. **Embedding Index**: All image embeddings are precomputed and stored.
4. **Retrieval**: Given a query/Image (e.g., *"bracelets"*), it is encoded, and top images are ranked via cosine similarity.
5. **UI**: Results are shown in a Streamlit web app with image grid and score annotations.

---
Sample search using CLIP embeddings:

Example result for Image Search:

<img width="746" height="793" alt="Screenshot 2026-01-21 at 5 10 37 PM" src="https://github.com/user-attachments/assets/f221a41b-e5d7-4551-b52f-87bda0f42964" />

Text Query result:

<img width="3600" height="2400" alt="results_bollywood style outfits" src="https://github.com/user-attachments/assets/eedee537-19af-44e1-948a-7299722966a3" />

