import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# === Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv("finetune_dataset.csv")
image_folder = "sample_images"

# === Train/Val Split ===
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)

# === Load model & processor ===
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Freeze all layers
for p in model.parameters():
    p.requires_grad = False

# Unfreeze last transformer layer + projection layers
for p in model.vision_model.encoder.layers[-1].parameters():
    p.requires_grad = True
for p in model.text_model.encoder.layers[-1].parameters():
    p.requires_grad = True
for p in model.visual_projection.parameters():
    p.requires_grad = True
for p in model.text_projection.parameters():
    p.requires_grad = True


# === Dataset ===
class CLIPContrastiveDataset(Dataset):
    def __init__(self, dataframe, image_folder, processor):
        self.df = dataframe
        self.image_folder = image_folder
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_folder, row['image'])
        text = str(row['description'])
        label = row['label']

        image = Image.open(img_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            truncation=True,
            max_length=77,
            padding=False
        )

        inputs["label"] = torch.tensor(label, dtype=torch.long)
        return inputs


# === Custom Collate Function ===
def collate_fn(batch):
    collated = {}

    collated["label"] = torch.stack([item["label"] for item in batch])
    collated["pixel_values"] = torch.stack([item["pixel_values"][0] for item in batch])

    token_batch = [
        {
            "input_ids": item["input_ids"][0],
            "attention_mask": item["attention_mask"][0]
        }
        for item in batch
    ]
    token_padded = processor.tokenizer.pad(token_batch, return_tensors="pt")
    collated["input_ids"] = token_padded["input_ids"]
    collated["attention_mask"] = token_padded["attention_mask"]

    return collated


# === Contrastive Loss ===
def clip_contrastive_loss(image_embeds, text_embeds):
    image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
    text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)
    logits = torch.matmul(image_embeds, text_embeds.T) * model.logit_scale.exp()

    labels = torch.arange(len(logits)).to(logits.device)
    return (torch.nn.functional.cross_entropy(logits, labels) +
            torch.nn.functional.cross_entropy(logits.T, labels)) / 2


# === Evaluation Loop ===
def evaluate(model, val_loader):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

            loss = clip_contrastive_loss(outputs.image_embeds, outputs.text_embeds)
            total_val_loss += loss.item()
    model.train()
    return total_val_loss / len(val_loader)


# === Dataloaders ===
train_dataset = CLIPContrastiveDataset(train_df, image_folder, processor)
val_dataset = CLIPContrastiveDataset(val_df, image_folder, processor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# === Optimizer ===
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

# === Training Loop ===
model.train()
epochs = 5

for epoch in range(epochs):
    total_train_loss = 0.0
    for batch in train_loader:
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        loss = clip_contrastive_loss(outputs.image_embeds, outputs.text_embeds)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    val_loss = evaluate(model, val_loader)

    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

# === Save Fine-tuned Model ===
model.save_pretrained("clip-finetuned")
processor.save_pretrained("clip-finetuned")
print("✅ Fine-tuned model saved to ./clip-finetuned/")