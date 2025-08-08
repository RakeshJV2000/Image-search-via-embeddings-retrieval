import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from rank_bm25 import BM25Okapi
import nltk
#nltk.download('punkt_tab')
nltk.data.path.append("/Users/rakeshjv/nltk_data")
from nltk.tokenize import word_tokenize


# === Load your dataset ===
df = pd.read_csv("data.csv")[["image", "description"]].dropna().drop_duplicates()
image_folder = "data"

# === Preprocess and tokenize descriptions ===
descriptions = df["description"].tolist()
tokenized_corpus = [word_tokenize(desc.lower()) for desc in descriptions]

# === Build BM25 index ===
bm25 = BM25Okapi(tokenized_corpus)

# === Accept query from user ===
query = "Black shirt".strip().lower()
tokenized_query = word_tokenize(query)

# === Get top 3 results ===
scores = bm25.get_scores(tokenized_query)
top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

# === Display results ===
print(f"\nTop matches for query: '{query}'\n")

plt.figure(figsize=(12, 4))

for i, idx in enumerate(top_indices):
    row = df.iloc[idx]
    image_name = row["image"]
    desc = row["description"]
    score = scores[idx]

    print(f"{i+1}. {image_name} (Score: {score:.2f})")
    print(f"   → {desc}\n")

    image_path = os.path.join(image_folder, image_name)
    img = Image.open(image_path).convert("RGB")

    plt.subplot(1, 3, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{image_name}\nScore: {score:.2f}", fontsize=9)

plt.suptitle(f"BM25 Top Matches for Query: {query}", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
