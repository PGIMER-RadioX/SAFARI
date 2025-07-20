#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import glob
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.foundation_model.training import train_foundation_model

def main():
    parser = argparse.ArgumentParser(description="Train a radiomics foundation model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # --- Data and Model Args ---
    parser.add_argument("--data", type=str, required=True, help="Directory or path to radiomics CSV file(s)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save model and artifacts. If None, a timestamped folder is created.")
    
    # --- Model Architecture Args ---
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    
    # --- Self-Supervised Learning Args ---
    parser.add_argument("--n-views", type=int, default=8, help="Number of masked views for contrastive learning")
    parser.add_argument("--mask-ratio", type=float, default=0.4, help="Mask ratio for self-supervised learning")
    parser.add_argument("--lambda", type=float, dest="lambda_weight", default=2.0, help="Weight for contrastive loss component")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for contrastive loss")

    # --- Training Hyperparameter Args ---
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for AdamW optimizer")
    parser.add_argument("--patience", type=int, default=15, help="Patience for early stopping on validation loss")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Value for gradient clipping to prevent exploding gradients")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Fraction of data to use for validation")
    parser.add_argument("--save-every", type=int, default=5, help="Save a model checkpoint every N epochs")

    # --- Other Args ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"radiomics_foundation_{timestamp}"
    
    if os.path.isdir(args.data):
        data_paths = glob.glob(os.path.join(args.data, "*.csv"))
    else:
        data_paths = [args.data]
    
    if not data_paths:
        print(f"Error: No CSV files found at path: {args.data}")
        return

    print(f"Starting training with {len(data_paths)} data file(s). Outputs will be saved to: {args.output_dir}")
    
    train_foundation_model(
        data_paths=data_paths, output_dir=args.output_dir, batch_size=args.batch_size,
        d_model=args.d_model, n_views=args.n_views, mask_ratio=args.mask_ratio,
        lambda_weight=args.lambda_weight, temperature=args.temperature, learning_rate=args.lr,
        max_epochs=args.epochs, weight_decay=args.weight_decay, early_stop_patience=args.patience,
        gradient_clip=args.gradient_clip, validation_split=args.validation_split,
        save_every=args.save_every, seed=args.seed
    )

if __name__ == "__main__":
    main()