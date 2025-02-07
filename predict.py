import torch
import torch.nn.functional as F
from src.model import KazClip
from src.text_encoder import TextTokenizer


def predict_top5(model, caption, image_embeddings, image_paths, device="cpu"):
    tokenizer = TextTokenizer()
    tokens = tokenizer(caption)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    image_embeddings = image_embeddings.to(device)

    with torch.no_grad():
        _, text_features = model(None, tokens)  

    text_features = F.normalize(text_features, dim=1)
    scores = text_features @ image_embeddings.t()
    top5_indices = scores.squeeze().topk(5).indices
    
    return [image_paths[i] for i in top5_indices]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_embeddings = torch.load("precomputed_image_embeddings.pt", map_location=device)
    image_paths = torch.load("image_paths.pt")

    model = KazClip()
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval().to(device)

    captions = ["көшеде келе жатқан адам", 
                "құсты жеп жатқан мысық", 
                "үстелде отырған мысық", 
                "үстелде отырған ит", 
                "үстелде отырған адам", 
                "үстелде отырған бала", 
                "бірге отырған ит пен мысық", 
                "көшеде келе жатқан адам және ит", 
                "терезенің алдында тұрған адам"]

    for caption in captions:
        print("Caption:", caption)
        top5 = predict_top5(model, caption, image_embeddings, image_paths, device=device)
        print("Top 5 matches:", top5)
