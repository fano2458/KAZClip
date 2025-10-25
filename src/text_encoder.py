import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("kz-transformers/kaz-roberta-conversational")

    def __call__(self, x):
        return self.tokenizer(x, return_tensors="pt", padding="max_length", max_length=64, truncation=True)


class TextEncoder(nn.Module):
    def __init__(self, projection_dim=256):
        super(TextEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained("kz-transformers/kaz-roberta-conversational")
        
        # Create deeper projection head with more trainable layers
        hidden_size = self.encoder.config.hidden_size  # Usually 768 for RoBERTa base
        intermediate_dim1 = hidden_size // 2  # 384
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

    def forward(self, x):
        out = self.encoder(**x)
        text_features = out.pooler_output
        projection = self.projection_head(text_features)

        return projection


if __name__ == "__main__":
    model = TextEncoder()
    tokenizer = AutoTokenizer.from_pretrained("kz-transformers/kz-roberta-conversational")
    input_text = "Сіздердің ата-аналарыңыздың аты-жөні туралы ақпарат беріңіз"
    input_ids = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=64, truncation=True)["input_ids"]
    # with torch.no_grad():
    #     output = model(input_ids)
    print(input_ids.shape)
