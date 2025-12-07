#!/usr/bin/env python3
"""
Inference script for KazClip - retrieve images based on Kazakh text queries.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import os
import argparse
from typing import List, Tuple
import json
from tqdm import tqdm

from src.model import KazClip
from src.text_encoder import TextTokenizer
from src.visual_encoder import VisualProcessor


class KazClipInference:
    """Inference wrapper for KazClip model."""
    
    def __init__(self, checkpoint_path: str, device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get('config', {})
        projection_dim = config.get('projection_dim', 512)
        self.visual_architecture = config.get('visual_architecture', 'deit_s_16')
        self.text_encoder_name = config.get('text_encoder_name', 'xlm-roberta-base')
        
        self.text_tokenizer = TextTokenizer(model_name=self.text_encoder_name)
        self.visual_processor = VisualProcessor(self.visual_architecture)
        
        self.model = KazClip(
            projection_dim=projection_dim,
            visual_architecture=self.visual_architecture,
            text_encoder_name=self.text_encoder_name,
            pretrained=False
        ).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Visual architecture: {self.visual_architecture}")
        print(f"Text encoder: {self.text_encoder_name}")
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            i2t = metrics.get('i2t_top1')
            t2i = metrics.get('t2i_top1')
            i2t_str = f"{i2t:.2f}%" if isinstance(i2t, (int, float)) else "N/A"
            t2i_str = f"{t2i:.2f}%" if isinstance(t2i, (int, float)) else "N/A"
            print(f"Model performance - I2T Top1: {i2t_str}, T2I Top1: {t2i_str}")
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text queries into embeddings."""
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                tokens = self.text_tokenizer(text)
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                
                text_features = self.model.encode_text(tokens)
                text_features = F.normalize(text_features, dim=1)
                embeddings.append(text_features.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def encode_images(self, image_paths: List[str]) -> torch.Tensor:
        """Encode images into embeddings."""
        embeddings = []
        
        with torch.no_grad():
            for image_path in tqdm(image_paths, desc="Encoding images"):
                try:
                    image = Image.open(image_path).convert("RGB")
                    image_tensor = self.visual_processor(image).to(self.device)
                    
                    visual_features = self.model.encode_image(image_tensor)
                    visual_features = F.normalize(visual_features, dim=1)
                    embeddings.append(visual_features.cpu())
                    
                except Exception as e:
                    print(f"Warning: Could not process {image_path}: {e}")
                    # Add zero embedding for failed images
                    if embeddings:
                        zero_embedding = torch.zeros_like(embeddings[0])
                    else:
                        zero_embedding = torch.zeros(1, self.model.projection_dim)
                    embeddings.append(zero_embedding)
        
        return torch.cat(embeddings, dim=0)
    
    def search_images(self, query: str, image_paths: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for images matching the text query."""
        # Encode query
        query_embedding = self.encode_text([query])
        
        # Encode images 
        image_embeddings = self.encode_images(image_paths)
        
        # Compute similarities
        similarities = query_embedding @ image_embeddings.t()
        similarities = similarities.squeeze(0)  # Remove batch dimension
        
        # Get top-k matches
        top_k = min(top_k, len(image_paths))
        top_indices = torch.topk(similarities, top_k).indices
        
        results = []
        for idx in top_indices:
            path = image_paths[idx.item()]
            score = similarities[idx.item()].item()
            results.append((path, score))
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Search images using Kazakh text queries')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image-dir', type=str, default='data/Images', help='Directory containing images')
    parser.add_argument('--query', type=str, help='Text query in Kazakh')
    parser.add_argument('--queries-file', type=str, help='File containing multiple queries (one per line)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top results to return')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--save-images', action='store_true', help='Copy result images to output directory')
    
    args = parser.parse_args()
    
    if not args.query and not args.queries_file:
        parser.error("Either --query or --queries-file must be specified")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize inference engine
    inference = KazClipInference(args.checkpoint, device)
    
    # Get image paths
    print(f"Loading images from {args.image_dir}")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    
    for filename in sorted(os.listdir(args.image_dir)):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(args.image_dir, filename))
    
    print(f"Found {len(image_paths)} images")
    
    # Prepare queries
    queries = []
    if args.query:
        queries = [args.query]
    elif args.queries_file:
        with open(args.queries_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process queries
    all_results = {}
    
    for query in queries:
        print(f"\nSearching for: '{query}'")
        results = inference.search_images(query, image_paths, args.top_k)
        
        print("Top results:")
        for i, (path, score) in enumerate(results, 1):
            filename = os.path.basename(path)
            print(f"  {i}. {filename} (similarity: {score:.4f})")
        
        all_results[query] = [(path, score) for path, score in results]
        
        # Save images if requested
        if args.save_images:
            query_dir = os.path.join(args.output_dir, f"query_{len(all_results)}")
            os.makedirs(query_dir, exist_ok=True)
            
            # Save query text
            with open(os.path.join(query_dir, 'query.txt'), 'w', encoding='utf-8') as f:
                f.write(query)
            
            # Copy images
            for i, (path, score) in enumerate(results, 1):
                filename = f"rank_{i:02d}_score_{score:.4f}_{os.path.basename(path)}"
                import shutil
                shutil.copy2(path, os.path.join(query_dir, filename))
    
    # Save results JSON
    results_file = os.path.join(args.output_dir, 'search_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {results_file}")
    if args.save_images:
        print(f"Images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
