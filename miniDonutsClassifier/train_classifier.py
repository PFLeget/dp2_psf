#!/usr/bin/env python
"""
CNN classifier for PSF quality (good vs bad).

Features:
- Small CNN with optional residual connection
- BCE loss with per-class weights (good/unsure/bad)
- Stratified batch sampling
- Learning rate scheduler (ReduceLROnPlateau)
- Tracks accuracy, precision, recall, F1, confusion matrix
- Early stopping based on validation recall
- Saves best model by recall (to catch all bad images)

Usage:
    python train_classifier.py --data training_data.h5 --epochs 50
"""

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import argparse
import json
from datetime import datetime


class PSFDataset(Dataset):
    """PyTorch Dataset for PSF stamps."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ResidualBlock(nn.Module):
    """Simple residual block."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ThisIsTheBadPsfClassifier(nn.Module):
    """
    Small CNN for 41x41 single-channel images.
    ~100k parameters.
    """

    def __init__(self, use_residual=True, dropout=0.0):
        super().__init__()

        self.use_residual = use_residual
        self.dropout_rate = dropout

        # Initial conv: 41x41x1 -> 39x39x32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(32)

        # Optional residual block
        self.res1 = ResidualBlock(32) if use_residual else nn.Identity()

        # 39x39x32 -> 19x19x32 (pool)
        self.pool1 = nn.MaxPool2d(2)

        # 19x19x32 -> 17x17x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(64)

        # 17x17x64 -> 8x8x64 (pool)
        self.pool2 = nn.MaxPool2d(2)

        # 8x8x64 -> 6x6x128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm2d(128)

        # 6x6x128 -> 3x3x128 (pool)
        self.pool3 = nn.MaxPool2d(2)

        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv blocks
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Classifier
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WeightedBCELoss(nn.Module):
    """
    BCE loss with per-sample weights based on class.
    Supports 3 classes: good (0), unsure (0.5), bad (1).
    """

    def __init__(self, weight_good=1.0, weight_unsure=1.0, weight_bad=1.0):
        super().__init__()
        self.weight_good = weight_good
        self.weight_unsure = weight_unsure
        self.weight_bad = weight_bad

    def forward(self, logits, targets):
        # Compute per-sample weights
        weights = torch.ones_like(targets)
        weights[targets == 0] = self.weight_good
        weights[(targets > 0) & (targets < 1)] = self.weight_unsure
        weights[targets == 1] = self.weight_bad

        # BCE loss with logits
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=weights, reduction='mean'
        )
        return bce


def compute_metrics(y_true, y_pred_prob, thresh_good=0.3, thresh_bad=0.7):
    """
    Compute classification metrics with 3-class predictions.

    Excludes true UNSURE samples (y=0.5) from metrics.

    Prediction classes:
    - Good: P < thresh_good (0.3)
    - Unsure: thresh_good <= P <= thresh_bad
    - Bad: P > thresh_bad (0.7)

    For true good/bad labels, predicting unsure is a misclassification.
    """
    # Exclude true unsure samples from metrics
    mask = (y_true == 0) | (y_true == 1)
    y_true_filtered = y_true[mask].astype(int)
    y_pred_prob_filtered = y_pred_prob[mask]

    n_unsure_excluded = int(np.sum(~mask))

    if len(y_true_filtered) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'confusion_matrix_3class': np.zeros((2, 3), dtype=int),
            'n_unsure_excluded': n_unsure_excluded,
        }

    # 3-class predictions: 0=good, 1=unsure, 2=bad
    y_pred_3class = np.ones_like(y_pred_prob_filtered, dtype=int)  # default unsure
    y_pred_3class[y_pred_prob_filtered < thresh_good] = 0  # good
    y_pred_3class[y_pred_prob_filtered > thresh_bad] = 2   # bad

    # Confusion matrix: rows=true (good=0, bad=1), cols=pred (good=0, unsure=1, bad=2)
    cm_3class = np.zeros((2, 3), dtype=int)
    for t in [0, 1]:
        for p in [0, 1, 2]:
            cm_3class[t, p] = np.sum((y_true_filtered == t) & (y_pred_3class == p))

    # Accuracy: correct only if (true=0 AND pred=good) OR (true=1 AND pred=bad)
    correct = ((y_true_filtered == 0) & (y_pred_3class == 0)) | \
              ((y_true_filtered == 1) & (y_pred_3class == 2))
    accuracy = np.mean(correct)

    # Precision for bad: TP / (TP + FP) where pred=bad
    # TP = true=1 AND pred=bad, FP = true=0 AND pred=bad
    tp = cm_3class[1, 2]
    fp = cm_3class[0, 2]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall for bad: TP / (TP + FN) where FN = true=1 AND pred!=bad
    # FN = true=1 AND (pred=good OR pred=unsure)
    fn = cm_3class[1, 0] + cm_3class[1, 1]
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix_3class': cm_3class,
        'n_unsure_excluded': n_unsure_excluded,
    }


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_targets = []
    all_probs = []

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_targets), np.array(all_probs))
    metrics['loss'] = avg_loss

    return metrics


def evaluate(model, loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X).squeeze(1)
            loss = criterion(logits, y)

            total_loss += loss.item() * X.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_targets), np.array(all_probs))
    metrics['loss'] = avg_loss

    return metrics


def create_stratified_sampler(y, oversample_minority=True):
    """
    Create a WeightedRandomSampler for stratified batches.
    Ensures each batch has proportional representation of classes.
    """
    # Compute class weights (inverse frequency)
    y_binary = (y >= 0.5).astype(int)  # unsure and bad are "positive"
    class_counts = np.bincount(y_binary)
    class_weights = 1.0 / class_counts

    # Assign weight to each sample
    sample_weights = class_weights[y_binary]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y),
        replacement=True
    )
    return sampler


def train(data_path, output_path="psf_classifier.pt", epochs=50, batch_size=128,
          lr=1e-3, weight_good=1.0, weight_unsure=2.0, weight_bad=10.0,
          val_split=0.1, patience=10, use_residual=True, dropout=0.0,
          seed=42, num_workers=4, verbose=True):
    """
    Train the PSF classifier.

    Can be called from CLI or notebook.

    Parameters
    ----------
    data_path : str
        Path to HDF5 training data file
    output_path : str
        Path to save the model
    epochs : int
        Maximum number of epochs
    batch_size : int
        Batch size
    lr : float
        Initial learning rate
    weight_good : float
        Loss weight for good (0) samples
    weight_unsure : float
        Loss weight for unsure (0.5) samples
    weight_bad : float
        Loss weight for bad (1) samples
    val_split : float
        Fraction of data for validation
    patience : int
        Early stopping patience
    use_residual : bool
        Use residual connection in model
    dropout : float
        Dropout rate (0 = no dropout)
    seed : int
        Random seed
    num_workers : int
        Number of data loader workers
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Training results with model, history, and best metrics
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Store hyperparameters
    hyperparams = {
        'data_path': data_path,
        'output_path': output_path,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_good': weight_good,
        'weight_unsure': weight_unsure,
        'weight_bad': weight_bad,
        'val_split': val_split,
        'patience': patience,
        'use_residual': use_residual,
        'dropout': dropout,
        'seed': seed,
    }

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    # Load data
    if verbose:
        print(f"\nLoading data from {data_path}")
    with h5py.File(data_path, 'r') as f:
        X = f['X'][:]
        y = f['y'][:]

    if verbose:
        print(f"Total samples: {len(X)}")
        print(f"Class distribution: good={np.sum(y == 0)}, unsure={np.sum(y == 0.5)}, bad={np.sum(y == 1)}")

    # Stratified train/val split
    y_stratify = (y >= 0.5).astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, stratify=y_stratify, random_state=seed
    )
    if verbose:
        print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")
        print(f"Train class dist: good={np.sum(y_train == 0)}, unsure={np.sum(y_train == 0.5)}, bad={np.sum(y_train == 1)}")
        print(f"Val class dist:   good={np.sum(y_val == 0)}, unsure={np.sum(y_val == 0.5)}, bad={np.sum(y_val == 1)}")

    # Create datasets
    train_dataset = PSFDataset(X_train, y_train)
    val_dataset = PSFDataset(X_val, y_val)

    # Stratified sampler for training
    train_sampler = create_stratified_sampler(y_train)

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Model
    model = ThisIsTheBadPsfClassifier(use_residual=use_residual, dropout=dropout).to(device)
    if verbose:
        print(f"\nModel parameters: {model.count_parameters():,}")

    # Loss and optimizer
    criterion = WeightedBCELoss(
        weight_good=weight_good,
        weight_unsure=weight_unsure,
        weight_bad=weight_bad
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=verbose
    )

    # Training loop
    if verbose:
        print(f"\nTraining with weights: good={weight_good}, unsure={weight_unsure}, bad={weight_bad}")
        print("=" * 100)

    best_f1 = 0
    best_metrics = {}
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Update scheduler based on F1 (we want to maximize F1)
        scheduler.step(val_metrics['f1'])

        # Log
        if verbose:
            print(f"Epoch {epoch+1:3d} | "
                  f"Train: loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} "
                  f"prec={train_metrics['precision']:.4f} rec={train_metrics['recall']:.4f} f1={train_metrics['f1']:.4f} | "
                  f"Val: loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
                  f"prec={val_metrics['precision']:.4f} rec={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f}")

            # Print confusion matrix every 10 epochs
            # Rows: true (good=0, bad=1), Cols: pred (good, unsure, bad)
            if (epoch + 1) % 10 == 0:
                cm = val_metrics['confusion_matrix_3class']
                print(f"         Val Confusion Matrix (true\\pred): good|unsure|bad")
                print(f"           True Good: {cm[0,0]:4d} | {cm[0,1]:4d} | {cm[0,2]:4d}")
                print(f"           True Bad:  {cm[1,0]:4d} | {cm[1,1]:4d} | {cm[1,2]:4d}")

        # Save history
        history.append({
            'epoch': epoch + 1,
            'train': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in train_metrics.items()},
            'val': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in val_metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
        })

        # Check for best model (based on F1-score)
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_metrics = val_metrics.copy()
            patience_counter = 0

            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
                'val_precision': val_metrics['precision'],
                'val_accuracy': val_metrics['accuracy'],
                'hyperparams': hyperparams,
            }, output_path)
            if verbose:
                print(f"         -> Saved best model (f1={val_metrics['f1']:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"\nEarly stopping after {patience} epochs without improvement")
            break

    if verbose:
        print("=" * 100)
        print(f"\nTraining complete!")
        print(f"Best validation F1: {best_f1:.4f}")
        print(f"Model saved to {output_path}")

    # Save training history with hyperparameters
    history_file = output_path.replace('.pt', '_history.json')
    results = {
        'hyperparams': hyperparams,
        'best_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in best_metrics.items()},
        'history': history,
    }
    with open(history_file, 'w') as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"Training history saved to {history_file}")

    # Return results for notebook use
    return {
        'model': model,
        'history': history,
        'best_metrics': best_metrics,
        'hyperparams': hyperparams,
        'device': device,
    }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Train PSF quality classifier")
    parser.add_argument("--data", type=str, required=True, help="HDF5 training data file")
    parser.add_argument("--output", type=str, default="psf_classifier.pt", help="Output model file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_good", type=float, default=1.0, help="Loss weight for good (0)")
    parser.add_argument("--weight_unsure", type=float, default=2.0, help="Loss weight for unsure (0.5)")
    parser.add_argument("--weight_bad", type=float, default=10.0, help="Loss weight for bad (1)")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--use_residual", action="store_true", default=True,
                       help="Use residual connection")
    parser.add_argument("--no_residual", dest="use_residual", action="store_false")
    parser.add_argument("--dropout", type=float, default=0.0,
                       help="Dropout rate (0 = no dropout)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_good=args.weight_good,
        weight_unsure=args.weight_unsure,
        weight_bad=args.weight_bad,
        val_split=args.val_split,
        patience=args.patience,
        use_residual=args.use_residual,
        dropout=args.dropout,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
