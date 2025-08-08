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

image_folder = "data"

data = np.load("finetuned_image_embeddings.npz", allow_pickle=True)
image_embeddings = data["embeddings"]
print(len(image_embeddings))
image_ids = data["image_ids"]


# === Accept query and compute similarity ===
query = "shoes"

with torch.no_grad():
    text_inputs = processor(text=[query], return_tensors="pt", truncation=True, max_length=77).to(device)
    text_emb = model.get_text_features(**text_inputs)
    text_emb = torch.nn.functional.normalize(text_emb, dim=-1).cpu().numpy()

# === Compute cosine similarity ===
scores = cosine_similarity(text_emb, image_embeddings).flatten()
top_indices = scores.argsort()[::-1][:3]

# === Display results ===
print("\n🎯 Top 3 matched images:")
# Show top 3 results with query shown once above all
# Show top 3 results with query string as the main title
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

for i, idx in enumerate(top_indices):
    matched_image = image_ids[idx]
    score = scores[idx]
    image_path = os.path.join(image_folder, matched_image)
    img = Image.open(image_path).convert("RGB")

    print(f"{i+1}. {matched_image} — Score: {score:.4f}")

    axs[i].imshow(img)
    axs[i].axis("off")
    axs[i].set_title(f"{matched_image}\nScore: {score:.2f}", fontsize=9)

# Add query title
fig.suptitle(f"Top Matches for Query: {query}", fontsize=14)

# Fix layout to accommodate the suptitle
plt.tight_layout(rect=[0, 0, 1, 0.93])
# plt.savefig(f"results_{query}.png", dpi=300)# reserve space for suptitle
plt.show()


