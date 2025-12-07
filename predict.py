import argparse
from typing import List, Sequence

import torch
import torch.nn.functional as F

from src.model import KazClip
from src.text_encoder import TextTokenizer


def load_model(checkpoint_path: str, device: torch.device) -> KazClip:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    projection_dim = config.get('projection_dim', 512)
    visual_arch = config.get('visual_architecture', 'deit_s_16')
    text_encoder = config.get('text_encoder_name', 'xlm-roberta-base')

    model = KazClip(
        projection_dim=projection_dim,
        visual_architecture=visual_arch,
        text_encoder_name=text_encoder,
        pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def predict_topk(model: KazClip, tokenizer: TextTokenizer, caption: str,
                 image_embeddings: torch.Tensor, image_paths: Sequence[str],
                 top_k: int, device: torch.device) -> List[tuple]:
    tokens = tokenizer(caption)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=1)
        scores = text_features @ image_embeddings.t()
        top_indices = scores.squeeze(0).topk(top_k).indices.tolist()

    return [(image_paths[idx], scores.squeeze(0)[idx].item()) for idx in top_indices]


def main():
    parser = argparse.ArgumentParser(description='Retrieve top-K images for Kazakh captions using KazClip embeddings')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--image-embeddings', type=str, required=True, help='Tensor file with precomputed image embeddings')
    parser.add_argument('--image-paths', type=str, required=True, help='Tensor/pt file with serialized image paths list')
    parser.add_argument('--captions', nargs='*', help='Captions to query (space separated)')
    parser.add_argument('--captions-file', type=str, help='File containing captions, one per line')
    parser.add_argument('--top-k', type=int, default=5, help='Number of images to retrieve')

    args = parser.parse_args()

    if not args.captions and not args.captions_file:
        parser.error('Provide at least one caption via --captions or --captions-file')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = load_model(args.checkpoint, device)
    tokenizer = TextTokenizer(model_name=config.get('text_encoder_name', 'xlm-roberta-base'))

    image_embeddings = torch.load(args.image_embeddings, map_location=device)
    image_paths = torch.load(args.image_paths)
    image_embeddings = F.normalize(image_embeddings.to(device), dim=1)

    captions: List[str] = []
    if args.captions:
        captions.extend(args.captions)
    if args.captions_file:
        with open(args.captions_file, 'r', encoding='utf-8') as f:
            captions.extend([line.strip() for line in f if line.strip()])

    for caption in captions:
        results = predict_topk(model, tokenizer, caption, image_embeddings, image_paths, args.top_k, device)
        print(f"\nCaption: {caption}")
        for rank, (path, score) in enumerate(results, start=1):
            print(f"  {rank}. {path} (similarity: {score:.4f})")


if __name__ == "__main__":
    main()
