import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import matplotlib.pyplot as plt
import os
import gc
import json
import joblib
from datetime import datetime
from tqdm import tqdm

from .model import CollaborativeModel, MaskedViewsDataset, DiscriminativeLoss

def train_foundation_model(
    data_paths, output_dir, batch_size=64, d_model=512, n_views=8, mask_ratio=0.4,
    lambda_weight=2.0, learning_rate=0.001, max_epochs=100, seed=42, validation_split=0.1,
    early_stop_patience=15, temperature=0.1, weight_decay=1e-5, gradient_clip=1.0, save_every=5
):
    """
    Train a foundation model for radiomics using self-supervised learning.
    This is the definitive, corrected training logic that faithfully reproduces the original's behavior.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'training_log.txt')
    
    def log_message(message):
        print(message)
        with open(log_path, 'a') as log_file:
            log_file.write(f"{message}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_message(f"Using device: {device}")
    
    all_features, all_metadata = [], []
    log_message(f"Loading data from {len(data_paths)} sources...")
    
    feature_names_from_first_file = None
    for path in tqdm(data_paths, desc="Loading data files"):
        df = pd.read_csv(path, low_memory=False)

        # BUG FIX: Re-implementing the EXACT logic from the original metadata_handler.py
        # This is the subtractive method: define metadata, features are what's left.
        metadata_patterns = [
            'patient', 'Patient', 'ID', 'id', 'mask', 'Mask', 'subvolume', 'grid', 
            'position', 'slice', 'target', 'Target', 'label', 'Label', 'class', 
            'Class', 'diagnostics'
        ]
        metadata_cols = []
        for col in df.columns:
            # Check for patterns that identify a column as metadata
            if any(pattern in col for pattern in metadata_patterns):
                metadata_cols.append(col)
            # Check for coordinate columns
            elif col.endswith('_i') or col.endswith('_j') or col.endswith('_k'):
                metadata_cols.append(col)
        
        # Features are all columns that are NOT metadata
        feature_cols = [c for c in df.columns if c not in metadata_cols]

        if feature_names_from_first_file is None:
            feature_names_from_first_file = feature_cols
        
        all_features.append(df[feature_names_from_first_file])
        all_metadata.append(df[metadata_cols])

    combined_features_df = pd.concat(all_features, ignore_index=True)
    combined_metadata_df = pd.concat(all_metadata, ignore_index=True)
    
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    combined_metadata_df.to_csv(metadata_path, index=False)
    log_message(f"Saved combined metadata for reference to {metadata_path}")
    
    X_all = combined_features_df.values.astype(np.float64)
    log_message(f"Combined dataset shape: {X_all.shape}")
    del all_features, all_metadata, combined_features_df, combined_metadata_df
    gc.collect()

    log_message("Calculating feature standard deviations for feature selection...")
    feature_std = np.std(X_all, axis=0)
    non_constant_cols_indices = np.where(feature_std > 1e-10)[0]
    X_all = X_all[:, non_constant_cols_indices]
    log_message(f"After removing constant features: {X_all.shape}")
    
    indices = np.arange(X_all.shape[0])
    np.random.shuffle(indices)
    split_idx = int(len(indices) * (1 - validation_split))
    X_train_raw, X_val_raw = X_all[indices[:split_idx]], X_all[indices[split_idx:]]
    del X_all
    gc.collect()
    
    log_message("Applying RobustScaler...")
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    joblib.dump(scaler, os.path.join(output_dir, 'robust_scaler.joblib'))
    
    log_message("Applying MinMaxScaler...")
    minmax = MinMaxScaler(feature_range=(-1, 1))
    X_train = minmax.fit_transform(X_train)
    X_val = minmax.transform(X_val)
    joblib.dump(minmax, os.path.join(output_dir, 'minmax_scaler.joblib'))

    log_message(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")
    
    log_message("Creating datasets...")
    train_dataset = MaskedViewsDataset(X_train, n_views=n_views, mask_ratio=mask_ratio)
    val_dataset = MaskedViewsDataset(X_val, n_views=n_views, mask_ratio=mask_ratio)
    
    log_message("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    input_size = X_train.shape[1]
    log_message(f"Creating model with input_size={input_size}, d_model={d_model}")
    model = CollaborativeModel(input_dim=input_size, d_model=d_model).to(device)
    log_message(f"Model structure:\n{model}")
    log_message(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    contrastive_loss_fn = DiscriminativeLoss(batch_size=batch_size, temperature=temperature)
    reconstruction_loss_fn = nn.MSELoss()

    config = {'input_dim': input_size, 'd_model': d_model, 'num_layers': 4}
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train':[], 'val':[], 'train_recon':[], 'val_recon':[], 'train_contrast':[], 'val_contrast':[]}

    log_message("Beginning training...")
    for epoch in range(max_epochs):
        model.train()
        train_loss_epoch, train_recon_epoch, train_contrast_epoch = 0.0, 0.0, 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")

        for x_views, x_original in progress_bar:
            x_views, x_original = x_views.to(device), x_original.to(device)
            batch_size_curr, n_views_curr, feat_dim = x_views.shape
            
            optimizer.zero_grad()
            
            x_views_flat = x_views.reshape(-1, feat_dim)
            embeddings, reconstructions = model(x_views_flat)
            
            embeddings = embeddings.reshape(batch_size_curr, n_views_curr, -1)
            
            contrast_loss_val = 0.0
            n_pairs = 0
            for i in range(n_views_curr):
                for j in range(i + 1, n_views_curr):
                    contrast_loss_val += contrastive_loss_fn(embeddings[:, i, :], embeddings[:, j, :])
                    n_pairs += 1
            contrast_loss = contrast_loss_val / n_pairs if n_pairs > 0 else 0.0

            recon_loss = reconstruction_loss_fn(reconstructions.view_as(x_views), x_original.unsqueeze(1).expand_as(x_views))
            
            loss = recon_loss + lambda_weight * contrast_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
            
            train_loss_epoch += loss.item()
            train_recon_epoch += recon_loss.item()
            train_contrast_epoch += contrast_loss.item() if isinstance(contrast_loss, torch.Tensor) else contrast_loss
            progress_bar.set_postfix({'loss': loss.item(), 'recon': recon_loss.item()})

        model.eval()
        val_loss_epoch, val_recon_epoch, val_contrast_epoch = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x_views, x_original in val_loader:
                x_views, x_original = x_views.to(device), x_original.to(device)
                batch_size_curr, n_views_curr, feat_dim = x_views.shape
                x_views_flat = x_views.reshape(-1, feat_dim)
                embeddings, reconstructions = model(x_views_flat)
                embeddings = embeddings.reshape(batch_size_curr, n_views_curr, -1)
                
                contrast_loss_val = 0.0
                n_pairs = 0
                for i in range(n_views_curr):
                    for j in range(i + 1, n_views_curr):
                        contrast_loss_val += contrastive_loss_fn(embeddings[:, i, :], embeddings[:, j, :])
                        n_pairs += 1
                contrast_loss = contrast_loss_val / n_pairs if n_pairs > 0 else 0.0
                
                recon_loss = reconstruction_loss_fn(reconstructions.view_as(x_views), x_original.unsqueeze(1).expand_as(x_views))
                loss = recon_loss + lambda_weight * contrast_loss
                
                val_loss_epoch += loss.item()
                val_recon_epoch += recon_loss.item()
                val_contrast_epoch += contrast_loss.item() if isinstance(contrast_loss, torch.Tensor) else contrast_loss

        avg_train_loss = train_loss_epoch / len(train_loader)
        avg_train_recon = train_recon_epoch / len(train_loader)
        avg_train_contrast = train_contrast_epoch / len(train_loader)
        history['train'].append(avg_train_loss)
        history['train_recon'].append(avg_train_recon)
        history['train_contrast'].append(avg_train_contrast)

        avg_val_loss = val_loss_epoch / len(val_loader)
        avg_val_recon = val_recon_epoch / len(val_loader)
        avg_val_contrast = val_contrast_epoch / len(val_loader)
        history['val'].append(avg_val_loss)
        history['val_recon'].append(avg_val_recon)
        history['val_contrast'].append(avg_val_contrast)
        
        log_message(
            f"Epoch {epoch+1}/{max_epochs}, "
            f"Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}, Contrast: {avg_train_contrast:.4f}), "
            f"Val Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon:.4f}, Contrast: {avg_val_contrast:.4f})"
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'config': config, 'best_val_loss': best_val_loss}, os.path.join(output_dir, 'best_model.pt'))
            log_message(f"Saved best model with val_loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % save_every == 0:
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict()}, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt'))

        if patience_counter >= early_stop_patience:
            log_message(f"Early stopping at epoch {epoch+1}")
            break

    log_message("Generating loss curve plot...")
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train'], label='Train Total Loss')
    plt.plot(history['val'], label='Validation Total Loss')
    plt.title('Total Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(history['train_recon'], label='Train Recon Loss')
    plt.plot(history['val_recon'], label='Validation Recon Loss')
    plt.title('Reconstruction Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_contrast'], label='Train Contrastive Loss')
    plt.plot(history['val_contrast'], label='Validation Contrastive Loss')
    plt.title('Contrastive Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    plt.close()

    for key, value in history.items():
        np.save(os.path.join(output_dir, f'{key}_losses.npy'), np.array(value))
    
    log_message(f"Training complete. Model and artifacts saved to: {output_dir}")
    return model, output_dir