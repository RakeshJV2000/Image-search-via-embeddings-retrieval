import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# === Load fine-tuned model ===
model_path = "clip-finetuned"
model = CLIPModel.from_pretrained(model_path).eval()
processor = CLIPProcessor.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# === Load image folder and get all unique image files ===
image_folder = "data"
all_images = sorted(set(os.listdir(image_folder)))  # ensures unique + sorted

# === Generate image embeddings ===
image_embeddings = []
image_ids = []

print("🔄 Generating image embeddings for unique files...")
i = 0
with torch.no_grad():
    for filename in all_images:
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue  # skip non-image files

        print(filename, "Currently processing")
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        image_emb = model.get_image_features(**inputs)
        image_emb = torch.nn.functional.normalize(image_emb, dim=-1)  # Normalize
        image_embeddings.append(image_emb.cpu().numpy().flatten())
        image_ids.append(filename)
        print("Count: ", i)
        i += 1

image_embeddings = np.vstack(image_embeddings)
print(f"Unique image embeddings shape: {image_embeddings.shape}")

# === Save as .npz file ===
np.savez("finetuned_image_embeddings.npz", embeddings=image_embeddings, image_ids=image_ids)
print("Saved to finetuned_image_embeddings.npz")
