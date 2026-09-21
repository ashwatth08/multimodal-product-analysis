import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import pandas as pd
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> Using device: {DEVICE}")

# 1. Dataset Class
class MultimodalDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None, max_len=64):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)

        encoded = self.tokenizer(
            str(row["text"]),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["label"]), dtype=torch.long)
        }

# 2. Cross-Attention Multi-Task Architecture
class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes=3, embed_dim=128):
        super().__init__()
        
        # Vision Backbone: ResNet-34 up to layer4
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.img_backbone = nn.Sequential(*list(resnet.children())[:-2]) # Shape: (B, 512, 7, 7)
        self.img_proj = nn.Conv2d(512, embed_dim, kernel_size=1)
        
        # Text Backbone: DistilBERT
        self.text_backbone = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_proj = nn.Linear(768, embed_dim)
        
        # Cross-Attention Layer (Text queries image patches)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        # Independent Heads (Prevents modality collapse)
        self.img_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.text_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Fused Multimodal Head
        self.fused_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        B = images.size(0)
        
        # 1. Image feature extraction
        img_feats = self.img_proj(self.img_backbone(images)) # (B, embed_dim, 7, 7)
        img_head_logits = self.img_head(img_feats)
        
        # Reshape image patches into sequence: (B, 49, embed_dim)
        img_seq = img_feats.flatten(2).permute(0, 2, 1)

        # 2. Text feature extraction
        txt_raw = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        txt_seq = self.text_proj(txt_raw) # (B, Seq_len, embed_dim)
        cls_token = txt_seq[:, 0, :]
        text_head_logits = self.text_head(cls_token)

        # 3. Cross Attention: Text queries Image patches
        attn_out, _ = self.cross_attn(query=txt_seq, key=img_seq, value=img_seq)
        attn_pooled = attn_out[:, 0, :] # Pooled cross-attended vector
        
        # 4. Joint Multi-modal Head
        fused = torch.cat([cls_token, attn_pooled], dim=-1)
        fused_logits = self.fused_head(fused)

        return fused_logits, img_head_logits, text_head_logits

# 3. Execution Pipeline
def main():
    df = pd.read_csv("clean_mvsa_paired.csv")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_loader = DataLoader(MultimodalDataset(train_df, tokenizer, transform_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(MultimodalDataset(test_df, tokenizer, transform_test), batch_size=16, shuffle=False)

    model = MultimodalClassifier(num_classes=3, embed_dim=128).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 5
    print("\n--- Training Multimodal Model with Multi-Head Supervision ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = batch["image"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            fused_logits, img_logits, txt_logits = model(images, input_ids, mask)

            # Combined loss: forces both image and text branches to learn representations independently
            loss_fused = criterion(fused_logits, labels)
            loss_img = criterion(img_logits, labels)
            loss_txt = criterion(txt_logits, labels)
            total_loss = loss_fused + (0.5 * loss_img) + (0.5 * loss_txt)

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            preds = fused_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f} - Accuracy: {(correct/total)*100:.2f}%")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            fused_logits, _, _ = model(images, input_ids, mask)
            all_preds.extend(fused_logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=["Negative", "Neutral", "Positive"]))
    torch.save(model.state_dict(), "multimodal_model.pth")
    print("Saved upgraded model to multimodal_model.pth")

if __name__ == "__main__":
    main()