import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# === Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "clip-finetuned"
image_folder = "data"

# === Load model and processor ===
@st.cache_resource
def load_model_and_processor():
    model = CLIPModel.from_pretrained(model_path).eval().to(device)
    processor = CLIPProcessor.from_pretrained(model_path)
    return model, processor

model, processor = load_model_and_processor()

# === Load image embeddings ===
@st.cache_data
def load_embeddings():
    data = np.load("finetuned_image_embeddings.npz", allow_pickle=True)
    return data["embeddings"], data["image_ids"]

image_embeddings, image_ids = load_embeddings()

# === Streamlit App UI ===
st.title("Search your product")
# st.markdown("Enter a query and retrieve the most relevant images.")

query = st.text_input("Search:", "")

if query:
    with torch.no_grad():
        inputs = processor(text=[query], return_tensors="pt", truncation=True, max_length=77).to(device)
        text_emb = model.get_text_features(**inputs)
        text_emb = torch.nn.functional.normalize(text_emb, dim=-1).cpu().numpy()

    scores = cosine_similarity(text_emb, image_embeddings).flatten()
    top_indices = scores.argsort()[::-1][:6]  # Top 6 images

    st.markdown(f"### Results for: *{query}*")

    cols = st.columns(3)  # 3 images per row

    for i, idx in enumerate(top_indices):
        img_name = image_ids[idx]
        score = scores[idx]
        img_path = os.path.join(image_folder, img_name)

        with cols[i % 3]:
            st.image(Image.open(img_path), caption=f"{img_name} (Score: {score:.2f})", use_container_width=True)
