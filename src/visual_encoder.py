import torch
from transformers import AutoImageProcessor, ViTModel

from src.projection import ProjectionHead

_VISION_CONFIG = {
    'deit_s_16': {
        'hf_repo': 'facebook/deit-small-patch16-224',
        'image_size': 224,
        'hidden_size': 384,
    }
}


def _get_architecture_config(architecture: str) -> dict:
    if architecture not in _VISION_CONFIG:
        raise ValueError(f"Unsupported visual architecture: {architecture}. Allowed: {tuple(_VISION_CONFIG.keys())}")
    return _VISION_CONFIG[architecture]


class VisualProcessor:
    """Pre-process images for ViT/DeiT models using the matching HuggingFace processor."""

    def __init__(self, architecture: str = 'deit_s_16'):
        config = _get_architecture_config(architecture)
        self.architecture = architecture
        self.image_size = config['image_size']
        self._processor = AutoImageProcessor.from_pretrained(config['hf_repo'])

    def __call__(self, image):
        processed = self._processor(images=image, return_tensors="pt")
        return processed['pixel_values']


class VisualEncoder(torch.nn.Module):
    """Vision Transformer/DeiT encoder with a residual projection head."""

    def __init__(self, architecture: str = 'deit_s_16', projection_dim: int = 256, pretrained: bool = True):
        super().__init__()
        config = _get_architecture_config(architecture)
        self.architecture = architecture
        self.input_size = config['image_size']
        self.projection_dim = projection_dim
        hf_repo = config['hf_repo']

        self.encoder = ViTModel.from_pretrained(hf_repo) if pretrained else ViTModel.from_pretrained(hf_repo, ignore_mismatched_sizes=False)
        hidden_size = self.encoder.config.hidden_size
        self.projection_head = ProjectionHead(hidden_size, projection_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values=pixel_values)
        visual_features = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
        return self.projection_head(visual_features)

    def get_feature_dim(self) -> int:
        return self.encoder.config.hidden_size

    def get_architecture_info(self) -> dict:
        return {
            'architecture': self.architecture,
            'hidden_size': self.encoder.config.hidden_size,
            'input_size': self.input_size,
            'projection_dim': self.projection_dim,
            'hf_repo': _get_architecture_config(self.architecture)['hf_repo']
        }


def get_available_architectures() -> list:
    return list(_VISION_CONFIG.keys())


def get_architecture_info(architecture: str) -> dict:
    config = _get_architecture_config(architecture)
    return {
        'architecture': architecture,
        'hidden_size': config['hidden_size'],
        'input_size': config['image_size'],
        'projection_dim': 256,
        'hf_repo': config['hf_repo'],
        'type': 'transformer'
    }