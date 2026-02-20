# 🔍 Visual Fashion Search Engine  

### Find the Outfit. Even When You Don’t Know Its Name.

Ever seen an outfit on Instagram or social media and thought:

> “I need this… but I have no idea what it’s called.”

No brand name.  
No product ID.  
No keywords.  
Just a screenshot.

That’s exactly the problem this project solves.

## 💡 The Problem

Traditional e-commerce search depends heavily on keyword matching and manually tagged metadata.

But fashion is visual.

Users don’t search for:
> “Asymmetrical ruched bodycon midi dress with sheer overlay.”

They search with vibes:

- “Bollywood-style outfit”
- “Cozy winter look”
- “Something like this screenshot”
- A screenshot from Instagram or Pinterest

Product descriptions are often incomplete, inconsistent, or misaligned with how users express visual intent.

---

## 🚀 The Solution

This project implements a **CLIP-powered visual search engine** that understands fashion the way people experience it — visually.

You can:

- 🖼️ **Upload an image → Find similar products**
- 📝 **Enter a text query → Retrieve visually relevant items**
- 🔎 Use **cosine similarity between embeddings** for accurate matching

The model is fine-tuned on custom fashion descriptions to better align visual features with real-world fashion language.

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

