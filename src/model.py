import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.text_encoder import TextEncoder
from src.visual_encoder import VisualEncoder, get_available_architectures

_ALLOWED_VISUAL = tuple(get_available_architectures())
_ALLOWED_TEXT = ('xlm-roberta-base',)

class KazClip(torch.nn.Module):
    def __init__(self, projection_dim: int = 512,
                 visual_architecture: str = 'deit_s_16',
                 text_encoder_name: str = 'xlm-roberta-base',
                 pretrained: bool = True,
                 **kwargs):
        super().__init__()
        if visual_architecture not in _ALLOWED_VISUAL:
            raise ValueError(f"Unsupported visual architecture: {visual_architecture}. Allowed: {_ALLOWED_VISUAL}")
        if text_encoder_name not in _ALLOWED_TEXT:
            raise ValueError(f"Unsupported text encoder: {text_encoder_name}. Allowed: {_ALLOWED_TEXT}")
        
        self.visual_architecture = visual_architecture
        self.text_encoder_name = text_encoder_name
        self.projection_dim = projection_dim
        
        self.visual_encoder = VisualEncoder(
            architecture=visual_architecture, 
            projection_dim=projection_dim, 
            pretrained=pretrained
        )
        self.text_encoder = TextEncoder(projection_dim, model_name=text_encoder_name)
        self.logit_scale = nn.Parameter(torch.ones(()) * math.log(1 / 0.07))
        
        print(f"🔧 KazClip initialized with:")
        print(f"   Visual: {visual_architecture}")
        print(f"   Projection dim: {projection_dim}")
        print(f"   Pretrained: {pretrained}")

    def forward(self, image, text):
        visual_features = self.encode_image(image) if image is not None else None
        text_features = self.encode_text(text) if text is not None else None
        return visual_features, text_features

    def encode_image(self, image):
        return self.visual_encoder(image)

    def encode_text(self, text):
        return self.text_encoder(text)

    def compute_loss(self, visual_features, text_features):
        visual_features = F.normalize(visual_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = visual_features @ text_features.t() * logit_scale
        batch_size = logits.shape[0]
        ground_truth = torch.arange(batch_size, device=logits.device)

        loss = (F.cross_entropy(logits, ground_truth) + F.cross_entropy(logits.t(), ground_truth)) / 2
        return loss
    
    def freeze_encoders(self, train_last_visual_layers: int = 2, train_last_text_layers: int = 0):
        """Freeze encoders while keeping projection heads and last transformer blocks trainable."""
        self._set_module_trainable(self.visual_encoder.encoder, requires_grad=False)
        self._set_module_trainable(self.text_encoder.encoder, requires_grad=False)

        self._unfreeze_last_layers(self._get_visual_layers(), train_last_visual_layers)
        self._unfreeze_last_layers(self._get_text_layers(), train_last_text_layers)

        self._set_module_trainable(self.visual_encoder.projection_head, requires_grad=True)
        self._set_module_trainable(self.text_encoder.projection_head, requires_grad=True)
        self.logit_scale.requires_grad = True

        print("🔒 Encoders frozen except for the requested tail layers")
        print(f"🔥 Visual encoder: last {train_last_visual_layers} layer(s) + projection head trainable")
        print(f"🔥 Text encoder: last {train_last_text_layers} layer(s) + projection head trainable")
        self._print_trainable_params()

    def _set_module_trainable(self, module: nn.Module, requires_grad: bool):
        for param in module.parameters():
            param.requires_grad = requires_grad

    def _unfreeze_last_layers(self, layers, count: int):
        if not layers or count <= 0:
            return
        for layer in layers[-count:]:
            self._set_module_trainable(layer, True)

    def _get_visual_layers(self):
        encoder = getattr(self.visual_encoder.encoder, 'encoder', None)
        if encoder is not None and hasattr(encoder, 'layer'):
            return list(encoder.layer)
        return []

    def _get_text_layers(self):
        encoder = getattr(self.text_encoder.encoder, 'encoder', None)
        if encoder is not None and hasattr(encoder, 'layer'):
            return list(encoder.layer)
        return []
    
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
            'text_encoder_name': self.text_encoder_name,
            'projection_dim': self.projection_dim,
            'visual_info': self.visual_encoder.get_architecture_info(),
            'text_info': {
                'hidden_size': self.text_encoder.encoder.config.hidden_size,
                'vocab_size': self.text_encoder.encoder.config.vocab_size,
            },
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'projection_details': {
                'visual_projection_params': visual_proj_params,
                'text_projection_params': text_proj_params,
                'projection_architecture': 'Residual MLP with LayerNorm',
                'total_projection_params': visual_proj_params + text_proj_params
            },
            'logit_scale': float(self.logit_scale.exp().item())
        }
