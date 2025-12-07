import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Multi-layer projection head with residual connections and normalization."""

    def __init__(self, input_dim: int, projection_dim: int, hidden_multiplier: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(int(projection_dim * hidden_multiplier), projection_dim)
        mid_dim = max(hidden_dim // 2, projection_dim)

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, projection_dim)
        )

        self.shortcut = nn.Linear(input_dim, projection_dim) if input_dim != projection_dim else nn.Identity()
        self.output_norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.net(x)
        residual = self.shortcut(x)
        combined = self.output_norm(projected + residual)
        return F.normalize(combined, dim=-1)
