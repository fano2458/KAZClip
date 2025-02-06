import torch
from torch.utils.data import Dataset, DataLoader
from src.model import KazClip
from src.visual_encoder import VisualProcessor
from src.text_encoder import TextTokenizer

import json
from PIL import Image
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import heapq
import os


class ImageCaptionDataset(Dataset):
    def __init__(self, data_path):
        self.visual_processor = VisualProcessor()
        self.text_tokenizer = TextTokenizer()

        with open(data_path, 'r') as f:
            data = json.load(f)
        self.labels = data['annotations']
        
        self.images = []
        self.captions = []

        ds_type = "train" if "train" in data_path else "val"
        for label in self.labels:
            image_path = f"data/{ds_type}2017/{label['image_id']:012d}.jpg"
            self.images.append(image_path)
            self.captions.append(label['caption'])

    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.visual_processor(image)[0,:]

        caption = self.captions[idx]
        tokens = self.text_tokenizer(caption)
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}

        return image, tokens


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KazClip().to(device)
    model.visual_encoder.encoder.requires_grad_(False)
    model.text_encoder.encoder.requires_grad_(False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {total_params - trainable_params}")

    valid_dataset = ImageCaptionDataset("data/val2017.json")
    train_dataset = ImageCaptionDataset("data/train2017.json")

    batch_size = 128
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 50
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    scaler = torch.cuda.amp.GradScaler()
    wandb.init(project="kaz-clip")
    best_val_loss = float('inf')
    patience = 4
    no_improvement_count = 0
    top_checkpoints = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        for batch_images, batch_text in tqdm(train_dataloader):
            batch_images = batch_images.to(device)
            batch_text = {k: v.to(device) for k, v in batch_text.items()}
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                visual_features, text_features = model(batch_images, batch_text)
                loss = model.compute_loss(visual_features, text_features)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
            train_batches += 1

        mean_train_loss = total_train_loss / train_batches
        print(f"Epoch {epoch+1}/{epochs} - Mean Train Loss: {mean_train_loss:.4f}")

        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_images, batch_text in tqdm(valid_dataloader):
                batch_images = batch_images.to(device)
                batch_text = {k: v.to(device) for k, v in batch_text.items()}
                with torch.cuda.amp.autocast():
                    visual_features, text_features = model(batch_images, batch_text)
                    loss = model.compute_loss(visual_features, text_features)
                total_val_loss += loss.item()
                val_batches += 1

        mean_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Mean Val Loss: {mean_val_loss:.4f}")
        
        wandb.log({"epoch": epoch+1, "train_loss": mean_train_loss, "val_loss": mean_val_loss})
        scheduler.step()
        
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_model_path = f"model_epoch_{epoch+1}_{mean_val_loss:.3f}.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved at epoch {epoch+1} with val loss: {best_val_loss:.4f}")
            no_improvement_count = 0
            # Maintain top 3 model.state_dicts
            heapq.heappush(top_checkpoints, (mean_val_loss, best_model_path))
            if len(top_checkpoints) > 3:
                _, old_checkpoint = heapq.heappop(top_checkpoints)
                os.remove(old_checkpoint)
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


if __name__ == "__main__":
    train_model()
