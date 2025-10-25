from src.text_encoder import TextEncoder
from src.visual_encoder import VisualEncoder, get_available_architectures

import torch
import torch.nn as nn
import torch.nn.functional as F


class KazClip(nn.Module):
    def __init__(self, projection_dim=256, visual_architecture='resnet50', pretrained=True):
        super(KazClip, self).__init__()
        
        self.visual_architecture = visual_architecture
        self.projection_dim = projection_dim
        
        self.visual_encoder = VisualEncoder(
            architecture=visual_architecture, 
            projection_dim=projection_dim, 
            pretrained=pretrained
        )
        self.text_encoder = TextEncoder(projection_dim)
        self.logit_scale = nn.Parameter(torch.ones(()))
        
        print(f"🔧 KazClip initialized with:")
        print(f"   Visual: {visual_architecture}")
        print(f"   Projection dim: {projection_dim}")
        print(f"   Pretrained: {pretrained}")

    def forward(self, image, text):
        visual_features = None
        text_features = None
        if image is not None:
            visual_features = self.visual_encoder(image)
        if text is not None:
            text_features = self.text_encoder(text)

        return visual_features, text_features

    def compute_loss(self, visual_features, text_features):
        visual_features = F.normalize(visual_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Compute logits with scaling
        logits = visual_features @ text_features.t() * torch.exp(self.logit_scale)
        batch_size = logits.shape[0]
        ground_truth = torch.arange(batch_size, device=logits.device)

        loss = (F.cross_entropy(logits, ground_truth) + F.cross_entropy(logits.t(), ground_truth)) / 2
        return loss
    
    def freeze_encoders(self):
        """Freeze visual encoder and text encoder parameters, keeping only projection layers trainable."""
        # Freeze visual encoder backbone
        for param in self.visual_encoder.encoder.parameters():
            param.requires_grad = False
        
        # Freeze text encoder backbone  
        for param in self.text_encoder.encoder.parameters():
            param.requires_grad = False
            
        # Keep projection heads trainable
        for param in self.visual_encoder.projection_head.parameters():
            param.requires_grad = True
            
        for param in self.text_encoder.projection_head.parameters():
            param.requires_grad = True
            
        # Keep logit scale trainable
        self.logit_scale.requires_grad = True
        
        print("🔒 Froze encoder backbones, keeping multi-layer projection heads trainable")
        print("🔥 Trainable components:")
        print("   - Visual projection head (4-layer MLP with BatchNorm)")
        print("   - Text projection head (4-layer MLP with BatchNorm)")
        print("   - Logit scale parameter")
        self._print_trainable_params()
    
    def _print_trainable_params(self):
        """Print information about trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"📊 Parameter Summary:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"   Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

    def get_model_info(self):
        """Return model configuration information."""
        # Get projection layer parameters counts
        visual_proj_params = sum(p.numel() for p in self.visual_encoder.projection_head.parameters())
        text_proj_params = sum(p.numel() for p in self.text_encoder.projection_head.parameters())
        
        return {
            'visual_architecture': self.visual_architecture,
            'projection_dim': self.projection_dim,
            'visual_info': self.visual_encoder.get_architecture_info(),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'projection_details': {
                'visual_projection_params': visual_proj_params,
                'text_projection_params': text_proj_params,
                'projection_architecture': '4-layer MLP with BatchNorm and Dropout',
                'total_projection_params': visual_proj_params + text_proj_params
            }
        }
