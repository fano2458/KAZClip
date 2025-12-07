#!/usr/bin/env python3
"""
Evaluation script for the trained KazClip model.
"""

import torch
from torch.utils.data import DataLoader
import argparse
import json
from train import KazakhImageCaptionDataset, evaluate_model
from src.model import KazClip


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> KazClip:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint or use defaults
    config = checkpoint.get('config', {})
    projection_dim = config.get('projection_dim', 512)
    visual_arch = config.get('visual_architecture', 'deit_s_16')
    text_encoder = config.get('text_encoder_name', 'xlm-roberta-base')
    
    model = KazClip(
        projection_dim=projection_dim,
        visual_architecture=visual_arch,
        text_encoder_name=text_encoder,
        pretrained=False  # Don't need pretrained for evaluation
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description='Evaluate KazClip model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--json-path', type=str, default='data/captions_kk_val2017.json', help='Path to COCO-style captions JSON')
    parser.add_argument('--image-dir', type=str, default='data/val2017', help='Path to images directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    config = checkpoint.get('config', {})
    
    # Create dataset
    print("Loading dataset...")
    dataset = KazakhImageCaptionDataset(
        json_path=args.json_path,
        image_dir=args.image_dir,
        visual_architecture=config.get('visual_architecture', 'deit_s_16'),
        text_model_name=config.get('text_encoder_name', 'xlm-roberta-base')
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    # Evaluate
    metrics = evaluate_model(model, dataloader, device)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"\nImage-to-Text Retrieval:")
    print(f"  Top-1: {metrics['i2t_top1']:.2f}%")
    print(f"  Top-5: {metrics['i2t_top5']:.2f}%")
    print(f"  Top-10: {metrics.get('i2t_top10', 'N/A')}")
    print(f"  Mean Rank: {metrics['i2t_mean_rank']:.2f}")
    print(f"  Median Rank: {metrics['i2t_median_rank']:.2f}")
    print(f"  MRR: {metrics['i2t_mrr']:.4f}")
    
    print(f"\nText-to-Image Retrieval:")
    print(f"  Top-1: {metrics['t2i_top1']:.2f}%")
    print(f"  Top-5: {metrics['t2i_top5']:.2f}%") 
    print(f"  Top-10: {metrics.get('t2i_top10', 'N/A')}")
    print(f"  Mean Rank: {metrics['t2i_mean_rank']:.2f}")
    print(f"  Median Rank: {metrics['t2i_median_rank']:.2f}")
    print(f"  MRR: {metrics['t2i_mrr']:.4f}")
    
    overall_score = (metrics['i2t_top1'] + metrics['i2t_top5'] + 
                    metrics['t2i_top1'] + metrics['t2i_top5']) / 4
    print(f"\nOverall Score: {overall_score:.2f}%")
    
    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'dataset_path': args.json_path,
        'dataset_size': len(dataset),
        'metrics': metrics,
        'overall_score': overall_score
    }
    
    if 'epoch' in checkpoint:
        results['epoch'] = checkpoint['epoch']
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
