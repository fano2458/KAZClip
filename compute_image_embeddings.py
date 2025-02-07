
import os
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from src.model import KazClip
from src.visual_encoder import VisualProcessor


def compute_image_embeddings(image_folder, model_path, output_path, device="cpu"):
    model = KazClip().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    processor = VisualProcessor()
    all_embeddings = []
    image_paths = []

    for fname in tqdm(os.listdir(image_folder)):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        path = os.path.join(image_folder, fname)
        image = Image.open(path).convert("RGB")
        image_tensor = processor(image)[0,:].unsqueeze(0).to(device)

        with torch.no_grad():
            visual_features, _ = model(image_tensor, None) 
        emb = F.normalize(visual_features, dim=1).squeeze(0).cpu()
        all_embeddings.append(emb)
        image_paths.append(path)

    image_embeddings = torch.stack(all_embeddings, dim=0)
    torch.save(image_embeddings, os.path.join(output_path, "precomputed_image_embeddings.pt"))
    torch.save(image_paths, os.path.join(output_path, "image_paths.pt"))

    return image_embeddings, image_paths


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_image_embeddings(
        image_folder="data/val2017",
        model_path="best_model.pt",
        output_path=".",
        device=device
    )