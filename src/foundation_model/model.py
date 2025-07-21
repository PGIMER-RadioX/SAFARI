# src/foundation_model/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class RadiomicsDataset(Dataset):
    """Dataset for radiomics features."""
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

class MaskedViewsDataset(Dataset):
    """Dataset that creates multiple masked views of each sample."""
    def __init__(self, features, n_views=8, mask_ratio=0.4):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.n_views = n_views
        self.mask_ratio = mask_ratio
        
    def __len__(self):
        return len(self.features)
    
    def create_masked_views(self, x):
        """Create n_views of the sample with different random masks."""
        views = []
        for _ in range(self.n_views):
            mask = torch.rand(x.shape) > self.mask_ratio
            x_masked = x.clone() * mask.float()
            views.append(x_masked)
        return torch.stack(views)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        x_views = self.create_masked_views(x)
        return x_views, x.clone()

class RadiomicsTransformer(nn.Module):
    def __init__(self, input_dim, d_model=512, num_layers=4, dropout=0.1):
        super(RadiomicsTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embeddings = nn.Parameter(torch.randn(1, input_dim, d_model))
        self.attention_layers = nn.ModuleList([
            SelfAttentionBlock(d_model, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2)
        )
        
    def forward(self, x, output_features=False):
        batch_size = x.shape[0]
        x = self.input_projection(x.unsqueeze(-1).transpose(1, 2))
        x = x + self.position_embeddings[:, :x.size(1), :]
        for layer in self.attention_layers:
            x = layer(x)
        x = self.layer_norm(x)
        x = torch.mean(x, dim=1)
        if output_features:
            return x
        else:
            return self.projection_head(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.BatchNorm1d(d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, output_dim)
        )
        
    def forward(self, x):
        return self.decoder(x)

class CollaborativeModel(nn.Module):
    def __init__(self, input_dim, d_model=512, num_layers=4):
        super(CollaborativeModel, self).__init__()
        self.encoder = RadiomicsTransformer(input_dim, d_model=d_model, num_layers=num_layers)
        self.decoder = Decoder(d_model, input_dim)
        
    def forward(self, x, output_features=False):
        features = self.encoder(x, output_features=True)
        if output_features:
            return features
        embedding = self.encoder.projection_head(features)
        reconstruction = self.decoder(features)
        return embedding, reconstruction

class DiscriminativeLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.1):
        super(DiscriminativeLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, z_i, z_j):
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss_i = self.criterion(sim_matrix, labels)
        loss_j = self.criterion(sim_matrix.T, labels)
        return (loss_i + loss_j) / 2.0