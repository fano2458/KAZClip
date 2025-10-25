import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import AutoModelForImageClassification, AutoImageProcessor, ViTModel, ViTImageProcessor
from typing import Union, Tuple
import warnings


# Supported architecture configurations
VISUAL_ARCHITECTURES = {
    # ResNet architectures
    'resnet34': {
        'model_name': 'resnet34',
        'hidden_size': 512,
        'input_size': 224,
        'type': 'torchvision'
    },
    'resnet50': {
        'model_name': 'resnet50', 
        'hidden_size': 2048,
        'input_size': 224,
        'type': 'torchvision'
    },
    'resnet101': {
        'model_name': 'resnet101',
        'hidden_size': 2048,
        'input_size': 224,
        'type': 'torchvision'
    },
    'resnet152': {
        'model_name': 'resnet152',
        'hidden_size': 2048,
        'input_size': 224,
        'type': 'torchvision'
    },
    
    # Vision Transformer architectures
    'vit-base': {
        'model_name': 'google/vit-base-patch16-224',
        'hidden_size': 768,
        'input_size': 224,
        'type': 'transformer'
    },
    'vit-large': {
        'model_name': 'google/vit-large-patch16-224',
        'hidden_size': 1024,
        'input_size': 224,
        'type': 'transformer'
    },
    
    # EfficientNet architectures
    'efficientnet-b0': {
        'model_name': 'efficientnet_b0',
        'hidden_size': 1280,
        'input_size': 224,
        'type': 'torchvision'
    },
    'efficientnet-b3': {
        'model_name': 'efficientnet_b3',
        'hidden_size': 1536,
        'input_size': 300,
        'type': 'torchvision'
    },
    'efficientnet-b5': {
        'model_name': 'efficientnet_b5',
        'hidden_size': 2048,
        'input_size': 456,
        'type': 'torchvision'
    },
    
    # Swin Transformer (original)
    'swinv2-base': {
        'model_name': 'microsoft/swinv2-base-patch4-window12-192-22k',
        'hidden_size': 1024,
        'input_size': 192,
        'type': 'swin'
    },
    'swinv2-large': {
        'model_name': 'microsoft/swinv2-large-patch4-window12-192-22k',
        'hidden_size': 1536,
        'input_size': 192,
        'type': 'swin'
    }
}


class VisualProcessor:
    """Unified processor for different visual architectures."""
    
    def __init__(self, architecture: str = 'resnet50'):
        if architecture not in VISUAL_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture: {architecture}. "
                           f"Supported: {list(VISUAL_ARCHITECTURES.keys())}")
        
        self.architecture = architecture
        self.config = VISUAL_ARCHITECTURES[architecture]
        self.input_size = self.config['input_size']
        
        if self.config['type'] == 'transformer':
            # Use HuggingFace processor for Vision Transformers
            self.processor = ViTImageProcessor.from_pretrained(self.config['model_name'])
        elif self.config['type'] == 'swin':
            # Use HuggingFace processor for Swin Transformers
            self.processor = AutoImageProcessor.from_pretrained(self.config['model_name'], use_fast=True)
        else:
            # Use torchvision transforms for ResNet and EfficientNet
            self.processor = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __call__(self, x):
        if self.config['type'] in ['transformer', 'swin']:
            return self.processor(x, return_tensors="pt")["pixel_values"]
        else:
            # For torchvision models, apply transforms directly
            if isinstance(x, list):
                return torch.stack([self.processor(img) for img in x])
            else:
                return self.processor(x).unsqueeze(0)


class VisualEncoder(nn.Module):
    """Unified visual encoder supporting multiple architectures."""
    
    def __init__(self, architecture: str = 'resnet50', projection_dim: int = 256, pretrained: bool = True):
        super(VisualEncoder, self).__init__()
        
        if architecture not in VISUAL_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture: {architecture}. "
                           f"Supported: {list(VISUAL_ARCHITECTURES.keys())}")
        
        self.architecture = architecture
        self.config = VISUAL_ARCHITECTURES[architecture]
        self.projection_dim = projection_dim
        
        # Load the appropriate model
        self.encoder = self._load_encoder(pretrained)
        
        # Create deeper projection head with more trainable layers
        hidden_size = self.config['hidden_size']
        intermediate_dim1 = hidden_size // 2
        intermediate_dim2 = max(projection_dim * 2, 512)  # Ensure reasonable intermediate size
        
        self.projection_head = nn.Sequential(
            # First projection layer
            nn.Linear(hidden_size, intermediate_dim1),
            nn.BatchNorm1d(intermediate_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            # Second projection layer
            nn.Linear(intermediate_dim1, intermediate_dim2),
            nn.BatchNorm1d(intermediate_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            # Third projection layer
            nn.Linear(intermediate_dim2, intermediate_dim2 // 2),
            nn.BatchNorm1d(intermediate_dim2 // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            # Final projection layer
            nn.Linear(intermediate_dim2 // 2, projection_dim)
        )
        
        print(f"✅ Loaded {architecture} with {self.config['hidden_size']} hidden size")
    
    def _load_encoder(self, pretrained: bool):
        """Load the appropriate encoder based on architecture type."""
        
        if self.config['type'] == 'torchvision':
            # Load torchvision models (ResNet, EfficientNet)
            model_func = getattr(models, self.config['model_name'])
            encoder = model_func(pretrained=pretrained)
            
            # Remove the final classification layer
            if 'resnet' in self.config['model_name']:
                encoder.fc = nn.Identity()
            elif 'efficientnet' in self.config['model_name']:
                encoder.classifier = nn.Identity()
            
            return encoder
            
        elif self.config['type'] == 'transformer':
            # Load Vision Transformer
            encoder = ViTModel.from_pretrained(self.config['model_name'])
            return encoder
            
        elif self.config['type'] == 'swin':
            # Load Swin Transformer
            encoder = AutoModelForImageClassification.from_pretrained(self.config['model_name'])
            encoder.classifier = nn.Identity()
            return encoder
        
        else:
            raise ValueError(f"Unknown model type: {self.config['type']}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the visual encoder."""
        
        if self.config['type'] == 'torchvision':
            # Standard torchvision models
            visual_features = self.encoder(x)
            
        elif self.config['type'] == 'transformer':
            # Vision Transformer
            outputs = self.encoder(x)
            visual_features = outputs.pooler_output
            
        elif self.config['type'] == 'swin':
            # Swin Transformer
            visual_features = self.encoder(x).logits
        
        # Apply projection head
        projection = self.projection_head(visual_features)
        return projection
    
    def get_feature_dim(self) -> int:
        """Return the feature dimension of the encoder."""
        return self.config['hidden_size']
    
    def get_architecture_info(self) -> dict:
        """Return information about the current architecture."""
        return {
            'architecture': self.architecture,
            'hidden_size': self.config['hidden_size'],
            'input_size': self.config['input_size'],
            'projection_dim': self.projection_dim,
            'type': self.config['type']
        }


def get_available_architectures() -> list:
    """Return list of available visual architectures."""
    return list(VISUAL_ARCHITECTURES.keys())


def get_architecture_info(architecture: str) -> dict:
    """Get detailed information about a specific architecture."""
    if architecture not in VISUAL_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return VISUAL_ARCHITECTURES[architecture].copy()


if __name__ == "__main__":
    # Test different architectures
    print("🧪 Testing visual encoders...")
    
    # Test architectures
    test_architectures = ['resnet50', 'vit-base', 'efficientnet-b0', 'swinv2-base']
    
    for arch in test_architectures:
        print(f"\n📋 Testing {arch}:")
        try:
            # Create processor and encoder
            processor = VisualProcessor(arch)
            encoder = VisualEncoder(arch, projection_dim=256, pretrained=False)  # Use pretrained=False for faster testing
            
            # Create dummy input
            from PIL import Image
            dummy_image = Image.new('RGB', (224, 224), color='red')
            
            # Process image
            if VISUAL_ARCHITECTURES[arch]['type'] in ['transformer', 'swin']:
                processed = processor(dummy_image)
            else:
                processed = processor(dummy_image)
            
            print(f"  Input shape: {processed.shape}")
            
            # Forward pass
            with torch.no_grad():
                output = encoder(processed)
            
            print(f"  Output shape: {output.shape}")
            print(f"  Architecture info: {encoder.get_architecture_info()}")
            print("  ✅ Success!")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n📊 Available architectures: {get_available_architectures()}")