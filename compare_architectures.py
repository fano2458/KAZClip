#!/usr/bin/env python3
"""
Architecture comparison script for KazClip.
Compare different visual encoders on the Kazakh dataset.
"""

import torch
import time
import psutil
import argparse
import pandas as pd
from pathlib import Path
from src.visual_encoder import get_available_architectures, VISUAL_ARCHITECTURES
from src.model import KazClip
from train import KazakhImageCaptionDataset, evaluate_model
from torch.utils.data import DataLoader
import json


def measure_model_performance(architecture: str, sample_size: int = 100) -> dict:
    """Measure computational performance of a model architecture."""
    
    print(f"📊 Testing {architecture}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Create model
        model = KazClip(
            visual_architecture=architecture,
            projection_dim=256,
            pretrained=False  # Faster loading for benchmarking
        ).to(device)
        
        # Get model info
        model_info = model.get_model_info()
        
        # Create sample dataset
        dataset = KazakhImageCaptionDataset(
            csv_path="data/captions_kazakh.txt",
            image_dir="data/Images",
            split="val",  # Use smaller validation set
            visual_architecture=architecture
        )
        
        # Limit sample size for faster testing
        indices = list(range(min(sample_size, len(dataset))))
        subset = torch.utils.data.Subset(dataset, indices)
        
        dataloader = DataLoader(subset, batch_size=8, shuffle=False, num_workers=2)
        
        # Memory before
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated(device) / 1024**2  # MB
        else:
            memory_before = 0
        
        # Time inference
        model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            for batch_images, batch_text in dataloader:
                batch_images = batch_images.to(device)
                batch_text = {k: v.to(device) for k, v in batch_text.items()}
                
                visual_features, text_features = model(batch_images, batch_text)
                
        end_time = time.time()
        
        # Memory after
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated(device) / 1024**2  # MB
            memory_used = memory_after - memory_before
        else:
            memory_used = 0
        
        inference_time = end_time - start_time
        samples_per_second = len(subset) / inference_time
        
        results = {
            'architecture': architecture,
            'success': True,
            'total_parameters': model_info['total_parameters'],
            'trainable_parameters': model_info['trainable_parameters'],
            'inference_time_seconds': inference_time,
            'samples_per_second': samples_per_second,
            'memory_used_mb': memory_used,
            'input_size': VISUAL_ARCHITECTURES[architecture]['input_size'],
            'hidden_size': VISUAL_ARCHITECTURES[architecture]['hidden_size'],
            'model_type': VISUAL_ARCHITECTURES[architecture]['type'],
            'sample_size': len(subset)
        }
        
        print(f"  ✅ {architecture}: {samples_per_second:.2f} samples/sec, {memory_used:.1f}MB")
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return results
        
    except Exception as e:
        print(f"  ❌ {architecture}: Error - {e}")
        return {
            'architecture': architecture,
            'success': False,
            'error': str(e),
            'sample_size': 0
        }


def run_architecture_comparison(architectures: list = None, sample_size: int = 100, output_file: str = "architecture_comparison.json"):
    """Run comparison across multiple architectures."""
    
    if architectures is None:
        architectures = get_available_architectures()
    
    print(f"🔄 Comparing {len(architectures)} architectures...")
    print(f"📊 Sample size: {sample_size}")
    print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    results = []
    
    for arch in architectures:
        result = measure_model_performance(arch, sample_size)
        results.append(result)
        
        # Small delay between tests
        time.sleep(2)
    
    # Create summary
    successful_results = [r for r in results if r.get('success', False)]
    
    if successful_results:
        print(f"\n📈 Performance Summary:")
        print(f"{'Architecture':<15} {'Params (M)':<10} {'Speed (s/s)':<12} {'Memory (MB)':<12} {'Input Size':<10}")
        print("-" * 65)
        
        for result in sorted(successful_results, key=lambda x: x['samples_per_second'], reverse=True):
            params_m = result['total_parameters'] / 1_000_000
            print(f"{result['architecture']:<15} {params_m:<10.1f} {result['samples_per_second']:<12.2f} "
                  f"{result['memory_used_mb']:<12.1f} {result['input_size']:<10}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    
    return results


def create_performance_report(results: list, output_file: str = "performance_report.md"):
    """Create a markdown performance report."""
    
    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', False)]
    
    with open(output_file, 'w') as f:
        f.write("# KazClip Visual Architecture Performance Report\n\n")
        
        f.write("## Test Configuration\n")
        f.write(f"- Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"- Sample size: {successful_results[0]['sample_size'] if successful_results else 'N/A'}\n")
        f.write(f"- Batch size: 8\n")
        f.write(f"- Projection dimension: 256\n\n")
        
        if successful_results:
            f.write("## Performance Results\n\n")
            f.write("| Architecture | Type | Params (M) | Speed (samples/sec) | Memory (MB) | Input Size | Hidden Size |\n")
            f.write("|--------------|------|------------|---------------------|-------------|------------|--------------|\n")
            
            for result in sorted(successful_results, key=lambda x: x['samples_per_second'], reverse=True):
                params_m = result['total_parameters'] / 1_000_000
                f.write(f"| {result['architecture']} | {result['model_type']} | {params_m:.1f} | "
                       f"{result['samples_per_second']:.2f} | {result['memory_used_mb']:.1f} | "
                       f"{result['input_size']} | {result['hidden_size']} |\n")
            
            # Speed ranking
            f.write("\n## Speed Ranking\n\n")
            for i, result in enumerate(sorted(successful_results, key=lambda x: x['samples_per_second'], reverse=True), 1):
                f.write(f"{i}. **{result['architecture']}**: {result['samples_per_second']:.2f} samples/sec\n")
            
            # Memory efficiency
            f.write("\n## Memory Efficiency\n\n")
            for i, result in enumerate(sorted(successful_results, key=lambda x: x['memory_used_mb']), 1):
                f.write(f"{i}. **{result['architecture']}**: {result['memory_used_mb']:.1f} MB\n")
        
        if failed_results:
            f.write("\n## Failed Tests\n\n")
            for result in failed_results:
                f.write(f"- **{result['architecture']}**: {result.get('error', 'Unknown error')}\n")
        
        f.write(f"\n## Recommendations\n\n")
        if successful_results:
            fastest = max(successful_results, key=lambda x: x['samples_per_second'])
            most_efficient = min(successful_results, key=lambda x: x['memory_used_mb'])
            
            f.write(f"- **Fastest**: {fastest['architecture']} ({fastest['samples_per_second']:.2f} samples/sec)\n")
            f.write(f"- **Most Memory Efficient**: {most_efficient['architecture']} ({most_efficient['memory_used_mb']:.1f} MB)\n")
            
            # Balance recommendation
            balanced = sorted(successful_results, key=lambda x: x['samples_per_second'] / (x['memory_used_mb'] + 1), reverse=True)[0]
            f.write(f"- **Best Balance**: {balanced['architecture']} (speed/memory ratio)\n")
    
    print(f"📄 Performance report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare KazClip visual architectures')
    parser.add_argument('--architectures', nargs='+', choices=get_available_architectures(),
                       help='Specific architectures to test (default: all)')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of samples to test (default: 100)')
    parser.add_argument('--output-json', type=str, default='architecture_comparison.json',
                       help='Output JSON file for results')
    parser.add_argument('--output-report', type=str, default='performance_report.md',
                       help='Output markdown report file')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with smaller sample size')
    
    args = parser.parse_args()
    
    if args.quick:
        args.sample_size = 20
        print("🏃 Running quick test...")
    
    architectures = args.architectures or get_available_architectures()
    
    print(f"🔍 Available architectures: {get_available_architectures()}")
    print(f"🎯 Testing architectures: {architectures}")
    
    # Run comparison
    results = run_architecture_comparison(
        architectures=architectures,
        sample_size=args.sample_size,
        output_file=args.output_json
    )
    
    # Create report
    create_performance_report(results, args.output_report)
    
    print("✅ Architecture comparison completed!")


if __name__ == "__main__":
    main()
