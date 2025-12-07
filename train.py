import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.model import KazClip
from src.visual_encoder import VisualProcessor, get_available_architectures
from src.text_encoder import TextTokenizer

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import wandb
import os
import argparse
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.simplefilter("ignore", FutureWarning)


class KazakhImageCaptionDataset(Dataset):
    """Dataset for Kazakh image-caption pairs with COCO/CSV support."""

    CAPTION_KEYS = ("caption", "caption_kk", "caption_kz")
    
    def __init__(self, csv_path: str = None, json_path: str = None, image_dir: str = None,
                 split: str = "train", test_size: float = 0.1,
                 random_state: int = 42, visual_architecture: str = 'deit_s_16',
                 text_model_name: str = 'xlm-roberta-base'):
        self.visual_processor = VisualProcessor(visual_architecture)
        try:
            self.text_tokenizer = TextTokenizer(model_name=text_model_name)
        except TypeError:
            # Backward compatibility if TextTokenizer does not accept model_name
            self.text_tokenizer = TextTokenizer()
        self.image_dir = image_dir
        self.data: List[Tuple[str, str]] = []

        if json_path is not None:
            self._load_from_coco_json(json_path)
        elif csv_path is not None:
            self._load_from_csv(csv_path, split, test_size, random_state)
        else:
            raise ValueError("Either json_path or csv_path must be provided.")

        if not self.data:
            raise ValueError(
                f"No valid image-caption pairs were loaded from {json_path or csv_path}."
            )
    
    def _load_from_coco_json(self, json_path: str):
        if not self.image_dir:
            raise ValueError("image_dir must be provided when loading from a COCO-style JSON file.")

        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        images = coco_data.get('images') or []
        annotations = coco_data.get('annotations') or []
        image_id_to_filename = {
            img['id']: img.get('file_name')
            for img in images
            if 'id' in img and img.get('file_name')
        }

        skipped_missing_caption = 0
        skipped_missing_image = 0

        for annotation in annotations:
            caption = self._extract_caption(annotation)
            if not caption:
                skipped_missing_caption += 1
                continue

            image_id = annotation.get('image_id')
            if image_id is None:
                skipped_missing_image += 1
                continue

            filename = image_id_to_filename.get(image_id)
            if not filename:
                filename = self._format_coco_filename(image_id)
                if not filename:
                    skipped_missing_image += 1
                    continue
            
            self.data.append((filename, caption))

        print(f"📊 Loaded {len(self.data)} image-caption pairs from {os.path.basename(json_path)}")
        if skipped_missing_caption:
            print(f"⚠️ Skipped {skipped_missing_caption} annotations without usable captions")
        if skipped_missing_image:
            print(f"⚠️ Missing filename metadata for {skipped_missing_image} annotations")

    def _load_from_csv(self, csv_path: str, split: str, test_size: float, random_state: int):
        # Load CSV format data (original implementation)
        df = pd.read_csv(csv_path)
        
        # Group by image to ensure no data leakage
        image_groups = df.groupby('image')['caption'].apply(list).reset_index()
        
        train_images, val_images = train_test_split(
            image_groups, test_size=test_size, random_state=random_state
        )
        
        selected_data = train_images if split == "train" else val_images
        
        for _, row in selected_data.iterrows():
            image_name = row['image']
            captions = row['caption']
            for caption in captions:
                self.data.append((image_name, caption))

    @classmethod
    def _extract_caption(cls, annotation: Dict) -> str:
        for key in cls.CAPTION_KEYS:
            value = annotation.get(key)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
        return None

    @staticmethod
    def _format_coco_filename(image_id) -> str:
        try:
            return f"{int(image_id):012d}.jpg"
        except (TypeError, ValueError):
            return None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_name, caption = self.data[idx]
        
        # Load and process image
        image_path = os.path.join(self.image_dir, image_name)
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.visual_processor(image)
            if len(image_tensor.shape) == 4:  # Remove batch dimension if present
                image_tensor = image_tensor[0, :]
        except Exception as e:
            # Return a black image if loading fails
            print(f"Warning: Could not load {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            image_tensor = self.visual_processor(image)
            if len(image_tensor.shape) == 4:  # Remove batch dimension if present
                image_tensor = image_tensor[0, :]
        
        # Tokenize caption
        tokens = self.text_tokenizer(caption)
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}
        
        return image_tensor, tokens


def compute_similarity_matrix(visual_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity matrix between visual and text features."""
    visual_features = F.normalize(visual_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    return visual_features @ text_features.t()


def compute_retrieval_metrics(similarity_matrix: torch.Tensor, k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute retrieval metrics for image-to-text and text-to-image retrieval.
    
    Args:
        similarity_matrix: (batch_size, batch_size) similarity matrix
        k_values: List of k values for top-k accuracy
        
    Returns:
        Dictionary containing various retrieval metrics
    """
    batch_size = similarity_matrix.size(0)
    
    # Image-to-text retrieval (rows are images, cols are texts)
    i2t_ranks = []
    for i in range(batch_size):
        # Get similarities for image i with all texts
        similarities = similarity_matrix[i]
        # Sort in descending order and get ranks
        sorted_indices = torch.argsort(similarities, descending=True)
        # Find rank of correct text (index i)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        i2t_ranks.append(rank)
    
    # Text-to-image retrieval (cols are images, rows are texts)  
    t2i_ranks = []
    for i in range(batch_size):
        # Get similarities for text i with all images
        similarities = similarity_matrix[:, i]
        # Sort in descending order and get ranks
        sorted_indices = torch.argsort(similarities, descending=True)
        # Find rank of correct image (index i)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        t2i_ranks.append(rank)
    
    i2t_ranks = np.array(i2t_ranks)
    t2i_ranks = np.array(t2i_ranks)
    
    metrics = {}
    
    # Top-k accuracy
    for k in k_values:
        metrics[f'i2t_top{k}'] = (i2t_ranks <= k).mean() * 100
        metrics[f't2i_top{k}'] = (t2i_ranks <= k).mean() * 100
    
    # Mean rank
    metrics['i2t_mean_rank'] = i2t_ranks.mean()
    metrics['t2i_mean_rank'] = t2i_ranks.mean()
    
    # Median rank
    metrics['i2t_median_rank'] = np.median(i2t_ranks)
    metrics['t2i_median_rank'] = np.median(t2i_ranks)
    
    # Mean reciprocal rank (MRR)
    metrics['i2t_mrr'] = (1.0 / i2t_ranks).mean()
    metrics['t2i_mrr'] = (1.0 / t2i_ranks).mean()
    
    # Recall at various k values
    for k in [1, 5, 10, 50]:
        if k <= batch_size:
            metrics[f'i2t_recall_at_{k}'] = (i2t_ranks <= k).mean() * 100
            metrics[f't2i_recall_at_{k}'] = (t2i_ranks <= k).mean() * 100
    
    return metrics


def evaluate_model(model: KazClip, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Comprehensive evaluation of the model on a validation dataset.
    
    Args:
        model: The KazClip model
        dataloader: DataLoader for validation data
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()
    
    all_visual_features = []
    all_text_features = []
    total_loss = 0.0
    num_batches = 0
    
    amp_enabled = device.type == 'cuda'

    with torch.no_grad():
        for batch_images, batch_text in tqdm(dataloader, desc="Evaluating"):
            batch_images = batch_images.to(device)
            batch_text = {k: v.to(device) for k, v in batch_text.items()}
            
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                visual_features, text_features = model(batch_images, batch_text)
                loss = model.compute_loss(visual_features, text_features)
            
            all_visual_features.append(visual_features.cpu())
            all_text_features.append(text_features.cpu())
            total_loss += loss.item()
            num_batches += 1
    
    # Concatenate all features
    all_visual_features = torch.cat(all_visual_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(all_visual_features, all_text_features)
    
    # Compute retrieval metrics
    metrics = compute_retrieval_metrics(similarity_matrix)
    
    # Add loss
    metrics['loss'] = total_loss / num_batches
    
    return metrics


def cleanup_root_checkpoints(visual_architecture: str):
    """Clean up old checkpoint files from the root directory (legacy cleanup)."""
    import glob
    
    # Get current working directory to search for legacy files
    current_dir = os.getcwd()
    
    # Find all old checkpoint files for this architecture in root directory
    pattern = os.path.join(current_dir, f"kazclip_{visual_architecture}_*.pt")
    old_checkpoints = glob.glob(pattern)
    
    if not old_checkpoints:
        # Also try pattern without architecture prefix (very old format)
        pattern = os.path.join(current_dir, f"kazclip_*.pt")
        old_checkpoints = glob.glob(pattern)
    
    for checkpoint in old_checkpoints:
        try:
            if os.path.exists(checkpoint):
                os.remove(checkpoint)
                print(f"🧹 Cleaned up legacy checkpoint: {os.path.basename(checkpoint)}")
        except OSError as e:
            print(f"⚠️ Could not remove {checkpoint}: {e}")
        except Exception as e:
            print(f"⚠️ Unexpected error removing {checkpoint}: {e}")
    
    if old_checkpoints:
        print(f"🧹 Legacy cleanup complete: processed {len(old_checkpoints)} files")


def load_existing_checkpoints(visual_architecture: str, max_checkpoints: int = 3) -> List[Tuple[float, str]]:
    """Load existing checkpoints for the given architecture and return top N."""
    checkpoint_dir = os.path.join('checkpoints', visual_architecture)
    
    if not os.path.exists(checkpoint_dir):
        return []
    
    import glob
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "kazclip_epoch_*.pt"))
    
    checkpoints_with_scores = []
    for checkpoint_path in checkpoint_files:
        try:
            # Try to load checkpoint and extract score
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            if 'overall_score' in checkpoint_data:
                score = checkpoint_data['overall_score']
            else:
                # Extract score from filename as fallback
                import re
                match = re.search(r'score_([\d.]+)', os.path.basename(checkpoint_path))
                score = float(match.group(1)) if match else 0.0
            
            checkpoints_with_scores.append((score, checkpoint_path))
        except Exception as e:
            print(f"⚠️ Could not load checkpoint {checkpoint_path}: {e}")
    
    # Sort by score and keep only top N
    checkpoints_with_scores.sort(reverse=True, key=lambda x: x[0])
    
    # Remove excess checkpoints
    if len(checkpoints_with_scores) > max_checkpoints:
        for _, excess_checkpoint in checkpoints_with_scores[max_checkpoints:]:
            try:
                if os.path.exists(excess_checkpoint):
                    os.remove(excess_checkpoint)
                    print(f"🗑️ Removed excess checkpoint: {os.path.basename(excess_checkpoint)}")
                else:
                    print(f"⚠️ Checkpoint not found for deletion: {excess_checkpoint}")
            except OSError as e:
                print(f"⚠️ Could not remove excess checkpoint {os.path.basename(excess_checkpoint)}: {e}")
            except Exception as e:
                print(f"⚠️ Unexpected error removing {os.path.basename(excess_checkpoint)}: {e}")
        checkpoints_with_scores = checkpoints_with_scores[:max_checkpoints]
    
    return checkpoints_with_scores


def save_training_history(history: Dict, checkpoint_dir: str):
    """Save training history to JSON file."""
    history_path = os.path.join(checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


def load_training_history(checkpoint_dir: str) -> Dict:
    """Load training history from JSON file."""
    history_path = os.path.join(checkpoint_dir, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            return json.load(f)
    return {'epochs': [], 'train_loss': [], 'val_loss': [], 'val_metrics': {}}


def update_training_graphs(history: Dict, checkpoint_dir: str, architecture: str):
    """Update and save training graphs for losses and metrics."""
    if not history['epochs']:
        return
    
    # Set matplotlib backend to avoid display issues
    plt.switch_backend('Agg')
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'KazClip Training Progress - {architecture.upper()}', fontsize=16, fontweight='bold')
    
    epochs = history['epochs']
    
    # 1. Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    if history['val_loss']:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # 2. Image-to-Text Retrieval
    if 'i2t_top1' in history['val_metrics'] and history['val_metrics']['i2t_top1']:
        ax2.plot(epochs, history['val_metrics']['i2t_top1'], 'g-', label='I2T Top-1', linewidth=2, marker='o', markersize=4)
        ax2.plot(epochs, history['val_metrics']['i2t_top5'], 'g--', label='I2T Top-5', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Image-to-Text Retrieval')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 100)
    
    # 3. Text-to-Image Retrieval
    if 't2i_top1' in history['val_metrics'] and history['val_metrics']['t2i_top1']:
        ax3.plot(epochs, history['val_metrics']['t2i_top1'], 'orange', label='T2I Top-1', linewidth=2, marker='o', markersize=4)
        ax3.plot(epochs, history['val_metrics']['t2i_top5'], 'orange', linestyle='--', label='T2I Top-5', linewidth=2, marker='s', markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Text-to-Image Retrieval')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)
    ax3.set_ylim(0, 100)
    
    # 4. Overall Score
    if 'overall_score' in history['val_metrics'] and history['val_metrics']['overall_score']:
        ax4.plot(epochs, history['val_metrics']['overall_score'], 'purple', label='Overall Score', linewidth=3, marker='D', markersize=6)
        # Mark the best score
        best_idx = np.argmax(history['val_metrics']['overall_score'])
        best_epoch = epochs[best_idx]
        best_score = history['val_metrics']['overall_score'][best_idx]
        ax4.plot(best_epoch, best_score, 'red', marker='*', markersize=12, label=f'Best: {best_score:.2f}%')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Overall Score (%)')
    ax4.set_title('Overall Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(left=0)
    ax4.set_ylim(0, 100)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(checkpoint_dir, 'training_progress.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Updated training graphs: {plot_path}")


def train_model(visual_architecture: str = 'deit_s_16', **kwargs):
    """Optimized training function for Kazakh CLIP model."""
    
    config = {
        'batch_size': 64,  
        'epochs': 100,
        'learning_rate': 5e-5,
        'weight_decay': 1e-4,
        'warmup_epochs': 5,
        'patience': 5,
        'projection_dim': 512,
        'max_checkpoints': 3,
        'eval_every_n_epochs': 1,
        'gradient_clip_val': 5.0,
        'random_state': 42,
        'visual_architecture': visual_architecture,
        'pretrained': True,
        'freeze_encoders': True,  # Default to freezing encoders for faster training
        'train_last_visual_layers': 2,
        'train_last_text_layers': 0,
        'train_json_path': 'data/captions_kk_train2017.json',
        'val_json_path': 'data/captions_kk_val2017.json',
        'train_image_dir': 'data/train2017',
        'val_image_dir': 'data/val2017',
        'text_encoder_name': 'xlm-roberta-base'
    }
    
    # Update config with any provided kwargs
    config.update(kwargs)
    
    print("🚀 Starting KazClip training...")
    print(f"Configuration: {config}")
    
    # Validate architecture
    if config['visual_architecture'] not in get_available_architectures():
        raise ValueError(f"Unsupported visual architecture: {config['visual_architecture']}. "
                        f"Available: {get_available_architectures()}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    amp_enabled = device.type == 'cuda'
    
    # Initialize model
    model_kwargs = {
        'projection_dim': config['projection_dim'],
        'visual_architecture': config['visual_architecture'],
        'pretrained': config['pretrained']
    }
    if config.get('text_encoder_name'):
        model_kwargs['text_encoder_name'] = config['text_encoder_name']
    try:
        model = KazClip(**model_kwargs).to(device)
    except TypeError:
        model_kwargs.pop('text_encoder_name', None)
        model = KazClip(**model_kwargs).to(device)
    
    # Print model info
    model_info = model.get_model_info()
    print(f"📊 Model Information:")
    for key, value in model_info.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"     {k}: {v}")
        else:
            print(f"   {key}: {value:,}" if isinstance(value, int) else f"   {key}: {value}")
    
    # Optionally freeze encoder backbones, keep only projection layers trainable
    if config.get('freeze_encoders', True):
        model.freeze_encoders(
            train_last_visual_layers=config.get('train_last_visual_layers', 2),
            train_last_text_layers=config.get('train_last_text_layers', 0)
        )
    else:
        print("🔥 Training all parameters (encoders not frozen)")
    
    # Create datasets
    print("📁 Loading datasets...")
    train_dataset = KazakhImageCaptionDataset(
        json_path=config.get('train_json_path', "data/captions_kk_train2017.json"),
        image_dir=config.get('train_image_dir', "data/train2017"),
        split="train",
        visual_architecture=config['visual_architecture'],
        text_model_name=config['text_encoder_name']
    )
    
    val_dataset = KazakhImageCaptionDataset(
        json_path=config.get('val_json_path', "data/captions_kk_val2017.json"),
        image_dir=config.get('val_image_dir', "data/val2017"),
        split="val",
        visual_architecture=config['visual_architecture'],
        text_model_name=config['text_encoder_name']
    )
    
    print(f"📊 Train dataset size: {len(train_dataset)}")
    print(f"📊 Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True  # For stable batch size
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Warmup + Cosine Annealing
    warmup_steps = max(1, len(train_loader) * config['warmup_epochs'])
    total_steps = max(warmup_steps + 1, len(train_loader) * config['epochs'])

    def lr_lambda(step: int) -> float:
        if step <= warmup_steps:
            return float(step) / float(warmup_steps)
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    
    # Initialize wandb
    if config.get('use_wandb', True):
        wandb.init(
            project=config.get('project_name', 'kaz-clip'),
            config=config,
            name=f"kazclip-{config['visual_architecture']}-{config['projection_dim']}d-{config['learning_rate']}"
        )
    
    # Training tracking
    best_metrics = {'i2t_top1': 0, 'i2t_top5': 0, 't2i_top1': 0, 't2i_top5': 0}
    best_overall_score = 0
    patience_counter = 0
    
    # Load existing checkpoints and clean up legacy ones
    saved_checkpoints = load_existing_checkpoints(config['visual_architecture'], config['max_checkpoints'])
    if saved_checkpoints:
        best_overall_score = saved_checkpoints[0][0]  # Get best score from existing checkpoints
        print(f"📂 Found {len(saved_checkpoints)} existing checkpoints, best score: {best_overall_score:.2f}%")
    
    # Clean up any old checkpoints from root directory
    cleanup_root_checkpoints(config['visual_architecture'])
    
    # Initialize training history tracking
    checkpoint_dir = os.path.join('checkpoints', config['visual_architecture'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load existing training history or create new one
    training_history = load_training_history(checkpoint_dir)
    
    print("🎯 Starting training loop...")
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch_idx, (batch_images, batch_text) in enumerate(progress_bar):
            batch_images = batch_images.to(device, non_blocking=True)
            batch_text = {k: v.to(device, non_blocking=True) for k, v in batch_text.items()}
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                visual_features, text_features = model(batch_images, batch_text)
                loss = model.compute_loss(visual_features, text_features)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_val'])
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log training metrics frequently
            if batch_idx % 50 == 0 and config.get('use_wandb', True):
                wandb.log({
                    'train_loss_step': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0],
                    'epoch': epoch + batch_idx / len(train_loader)
                })
        
        avg_train_loss = train_loss / train_batches
        print(f"📈 Epoch {epoch+1} - Average Train Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        if (epoch + 1) % config['eval_every_n_epochs'] == 0:
            print("🔍 Running validation...")
            val_metrics = evaluate_model(model, val_loader, device)
            
            # Calculate overall score (average of key metrics)
            overall_score = (
                val_metrics['i2t_top1'] + val_metrics['i2t_top5'] + 
                val_metrics['t2i_top1'] + val_metrics['t2i_top5']
            ) / 4
            
            # Print validation results
            print(f"📊 Validation Results - Epoch {epoch+1}:")
            print(f"   Loss: {val_metrics['loss']:.4f}")
            print(f"   I2T - Top1: {val_metrics['i2t_top1']:.2f}% | Top5: {val_metrics['i2t_top5']:.2f}%")
            print(f"   T2I - Top1: {val_metrics['t2i_top1']:.2f}% | Top5: {val_metrics['t2i_top5']:.2f}%")
            print(f"   Overall Score: {overall_score:.2f}%")
            
            # Update training history
            training_history['epochs'].append(epoch + 1)
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(val_metrics['loss'])
            
            # Initialize nested metrics dictionaries if they don't exist
            for metric_name in ['i2t_top1', 'i2t_top5', 't2i_top1', 't2i_top5', 'overall_score']:
                if metric_name not in training_history['val_metrics']:
                    training_history['val_metrics'][metric_name] = []
            
            # Add current metrics to history
            training_history['val_metrics']['i2t_top1'].append(val_metrics['i2t_top1'])
            training_history['val_metrics']['i2t_top5'].append(val_metrics['i2t_top5'])
            training_history['val_metrics']['t2i_top1'].append(val_metrics['t2i_top1'])
            training_history['val_metrics']['t2i_top5'].append(val_metrics['t2i_top5'])
            training_history['val_metrics']['overall_score'].append(overall_score)
            
            # Save training history and update graphs
            save_training_history(training_history, checkpoint_dir)
            update_training_graphs(training_history, checkpoint_dir, config['visual_architecture'])
            
            # Log to wandb
            if config.get('use_wandb', True):
                wandb_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': val_metrics['loss'],
                    'val_overall_score': overall_score,
                }
                wandb_metrics.update({f'val_{k}': v for k, v in val_metrics.items() if k != 'loss'})
                wandb.log(wandb_metrics)
            
            # Save best model
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_metrics.update({
                    'i2t_top1': val_metrics['i2t_top1'],
                    'i2t_top5': val_metrics['i2t_top5'], 
                    't2i_top1': val_metrics['t2i_top1'],
                    't2i_top5': val_metrics['t2i_top5']
                })
                
                # Create checkpoint directory structure
                checkpoint_dir = os.path.join('checkpoints', config['visual_architecture'])
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Create checkpoint filename with timestamp for uniqueness
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_filename = f"kazclip_epoch_{epoch+1}_score_{overall_score:.2f}_{timestamp}.pt"
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
                
                # Save checkpoint
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics': val_metrics,
                    'config': config,
                    'model_info': model.get_model_info(),
                    'overall_score': overall_score,
                    'visual_architecture': config['visual_architecture']
                }, checkpoint_path)
                
                print(f"💾 New best model saved: {checkpoint_path}")
                
                # Manage checkpoints - keep only top 3 for this architecture
                saved_checkpoints.append((overall_score, checkpoint_path))
                saved_checkpoints.sort(reverse=True, key=lambda x: x[0])
                
                # Remove old checkpoints beyond max_checkpoints
                while len(saved_checkpoints) > config['max_checkpoints']:
                    _, old_checkpoint = saved_checkpoints.pop()
                    try:
                        if os.path.exists(old_checkpoint):
                            os.remove(old_checkpoint)
                            print(f"🗑️ Removed old checkpoint: {os.path.basename(old_checkpoint)}")
                        else:
                            print(f"⚠️ Old checkpoint not found: {old_checkpoint}")
                    except OSError as e:
                        print(f"⚠️ Could not remove old checkpoint {os.path.basename(old_checkpoint)}: {e}")
                    except Exception as e:
                        print(f"⚠️ Unexpected error removing {os.path.basename(old_checkpoint)}: {e}")
                
                # Also clean up any old checkpoints from root directory (legacy cleanup)
                cleanup_root_checkpoints(config['visual_architecture'])
                
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= config['patience']:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break
        else:
            # Log training loss even when not evaluating
            if config.get('use_wandb', True):
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                })
    
    print("🎉 Training completed!")
    print(f"🏆 Best Overall Score: {best_overall_score:.2f}%")
    print(f"🏆 Best Metrics:")
    for metric, value in best_metrics.items():
        print(f"   {metric}: {value:.2f}%")
    
    if config.get('use_wandb', True):
        wandb.finish()
    return model, best_metrics


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Train KazClip with different visual architectures')
    
    # Model architecture options
    parser.add_argument('--visual-architecture', type=str, default='deit_s_16',
                       choices=get_available_architectures(),
                       help='Visual encoder architecture to use (currently only deit_s_16)')
    parser.add_argument('--projection-dim', type=int, default=256,
                       help='Projection dimension for embeddings')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Do not use pretrained weights')
    
    # Training hyperparameters
    parser.add_argument('--freeze-encoders', action='store_true',
                       help='Freeze encoder backbones, train only projection layers')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                       help='Number of warmup epochs')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--gradient-clip-val', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--train-last-visual-layers', type=int, default=2,
                       help='Number of final ViT blocks to keep trainable when freezing encoders')
    parser.add_argument('--train-last-text-layers', type=int, default=0,
                       help='Number of final text transformer blocks to keep trainable when freezing encoders')
    
    # Data options
    parser.add_argument('--train-json-path', type=str, default='data/captions_kk_train2017.json',
                       help='Path to training captions JSON file')
    parser.add_argument('--val-json-path', type=str, default='data/captions_kk_val2017.json',
                       help='Path to validation captions JSON file')
    parser.add_argument('--train-image-dir', type=str, default='data/train2017',
                       help='Directory containing training images')
    parser.add_argument('--val-image-dir', type=str, default='data/val2017',
                       help='Directory containing validation images')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    
    # Training options
    parser.add_argument('--eval-every-n-epochs', type=int, default=1,
                       help='Evaluate every N epochs')
    parser.add_argument('--max-checkpoints', type=int, default=3,
                       help='Maximum number of checkpoints to keep')
    
    # Wandb options
    parser.add_argument('--project-name', type=str, default='kaz-clip',
                       help='Wandb project name')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')
    
    args = parser.parse_args()
    
    print("🔍 Available visual architectures:")
    for arch in get_available_architectures():
        print(f"  - {arch}")
    print(f"\n🎯 Selected architecture: {args.visual_architecture}")
    
    # Convert args to config dictionary
    config = {
        'visual_architecture': args.visual_architecture,
        'projection_dim': args.projection_dim,
        'pretrained': not args.no_pretrained,
        'freeze_encoders': args.freeze_encoders,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'patience': args.patience,
        'gradient_clip_val': args.gradient_clip_val,
        'train_last_visual_layers': args.train_last_visual_layers,
        'train_last_text_layers': args.train_last_text_layers,
        'random_state': args.random_state,
        'eval_every_n_epochs': args.eval_every_n_epochs,
        'max_checkpoints': args.max_checkpoints,
        'train_json_path': args.train_json_path,
        'val_json_path': args.val_json_path,
        'train_image_dir': args.train_image_dir,
        'val_image_dir': args.val_image_dir,
        'project_name': args.project_name,
        'use_wandb': not args.no_wandb,
        'text_encoder_name': 'xlm-roberta-base',
    }
    
    try:
        model, metrics = train_model(**config)
        print("✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
