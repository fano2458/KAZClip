import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.projection import ProjectionHead

_SUPPORTED_TEXT_MODEL = 'xlm-roberta-base'


class TextTokenizer:
    """Tokenizer wrapper that enforces the supported multilingual model."""

    def __init__(self, model_name: str = _SUPPORTED_TEXT_MODEL, max_length: int = 64, **kwargs):
        if model_name != _SUPPORTED_TEXT_MODEL:
            raise ValueError(f"Unsupported text encoder: {model_name}. Allowed: {_SUPPORTED_TEXT_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __call__(self, text, **tokenizer_kwargs):
        if isinstance(text, str):
            text = [text]
        max_length = tokenizer_kwargs.pop('max_length', self.max_length)
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            **tokenizer_kwargs
        )


class TextEncoder(nn.Module):
    """XLM-RoBERTa encoder followed by a residual projection head."""

    def __init__(self, projection_dim: int = 256, model_name: str = _SUPPORTED_TEXT_MODEL):
        super().__init__()
        if model_name != _SUPPORTED_TEXT_MODEL:
            raise ValueError(f"Unsupported text encoder: {model_name}. Allowed: {_SUPPORTED_TEXT_MODEL}")
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.projection_head = ProjectionHead(hidden_size, projection_dim)

    def forward(self, tokens):
        outputs = self.encoder(**tokens)
        hidden_states = outputs.last_hidden_state
        attention_mask = tokens.get('attention_mask')
        if attention_mask is not None:
            pooled = self._mean_pool(hidden_states, attention_mask)
        else:
            pooled = hidden_states[:, 0]
        return self.projection_head(pooled)

    @staticmethod
    def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
        masked_states = hidden_states * mask
        summed = masked_states.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts
