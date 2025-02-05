import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor


class VisualProcessor:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    
    def __call__(self, x):
        return self.processor(x, return_tensors="pt")["pixel_values"]


class VisualEncoder(nn.Module):
    def __init__(self, projection_dim=256):
        super(VisualEncoder, self).__init__()
        self.encoder = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.projection_head = nn.Linear(self.encoder.config.hidden_size, projection_dim)

        self.encoder.classifier = nn.Identity()

    def forward(self, x):
        visual_features = self.encoder(x).logits
        projection = self.projection_head(visual_features)

        return projection


if __name__ == "__main__":
    model = VisualEncoder()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    print(output.shape)