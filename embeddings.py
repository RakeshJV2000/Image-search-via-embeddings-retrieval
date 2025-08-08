import os
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# === CONFIG ===
device = "cuda" if torch.cuda.is_available() else "cpu"
image_folder = "sample_images"
csv_path = "sampled_images.csv"

# === Load Data ===
df = pd.read_csv(csv_path)

# === Load CLIP Model ===
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === Containers for embeddings ===
image_embeddings = []
text_embeddings = []

# === Generate Embeddings ===
for _, row in tqdm(df.iterrows(), total=len(df)):
    image_path = os.path.join(image_folder, row['image'])
    description = row['description']

    try:
        # --- Process Image ---
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_emb = model.get_image_features(**inputs)
        image_emb = image_emb.cpu().numpy().flatten()

        # --- Process Description ---
        # text_inputs = processor(text=[description], return_tensors="pt").to(device)
        # --- Process Description ---
        description = str(row['description']) if pd.notna(row['description']) else ""
        text_inputs = processor(
            text=[description],
            return_tensors="pt",
            truncation=True,
            max_length=77  # CLIP's limit
        ).to(device)

        with torch.no_grad():
            text_emb = model.get_text_features(**text_inputs)

        with torch.no_grad():
            text_emb = model.get_text_features(**text_inputs)
        text_emb = text_emb.cpu().numpy().flatten()

        # --- Save ---
        image_embeddings.append(image_emb)
        text_embeddings.append(text_emb)

    except Exception as e:
        print(f"❌ Error with image {row['image']}: {e}")
        image_embeddings.append(None)
        text_embeddings.append(None)

# === Add to DataFrame ===
df['image_embedding'] = image_embeddings
df['description_embedding'] = text_embeddings

# === Keep only needed columns ===
emb_df = df[['image', 'image_embedding', 'description', 'description_embedding']]
print(emb_df.head())
# === Save to file (optional) ===
emb_df.to_pickle("clip_embeddings.pkl")  # Save with numpy arrays intact

print("✅ Done. Embeddings generated and saved.")