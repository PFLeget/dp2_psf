"""
Example notebook usage for PSF classifier training.

Copy this into a Jupyter notebook cell by cell.
"""

# %% Cell 1: Imports
import numpy as np
import matplotlib.pyplot as plt
import json

# Import training function and model
from train_classifier import train, ThisIsTheBadPsfClassifier, PSFDataset
import torch
import h5py

# %% Cell 2: Train the model
# Adjust hyperparameters as needed

results = train(
    data_path="training_data.h5",
    output_path="psf_classifier.pt",
    epochs=50,
    batch_size=128,
    lr=1e-3,
    weight_good=1.0,
    weight_unsure=2.0,
    weight_bad=10.0,
    val_split=0.1,
    patience=10,
    use_residual=True,
    dropout=0.0,  # No dropout by default
    seed=42,
    num_workers=0,  # Use 0 for notebooks to avoid multiprocessing issues
    verbose=True,
)

# %% Cell 3: Plot training curves
history = results['history']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss
ax = axes[0, 0]
ax.plot([h['epoch'] for h in history], [h['train']['loss'] for h in history], label='Train')
ax.plot([h['epoch'] for h in history], [h['val']['loss'] for h in history], label='Val')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Loss')
ax.legend()
ax.grid(True)

# Accuracy
ax = axes[0, 1]
ax.plot([h['epoch'] for h in history], [h['train']['accuracy'] for h in history], label='Train')
ax.plot([h['epoch'] for h in history], [h['val']['accuracy'] for h in history], label='Val')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy')
ax.legend()
ax.grid(True)

# Recall (most important for us)
ax = axes[1, 0]
ax.plot([h['epoch'] for h in history], [h['train']['recall'] for h in history], label='Train')
ax.plot([h['epoch'] for h in history], [h['val']['recall'] for h in history], label='Val')
ax.set_xlabel('Epoch')
ax.set_ylabel('Recall')
ax.set_title('Recall (catching bad images)')
ax.legend()
ax.grid(True)

# F1
ax = axes[1, 1]
ax.plot([h['epoch'] for h in history], [h['train']['f1'] for h in history], label='Train')
ax.plot([h['epoch'] for h in history], [h['val']['f1'] for h in history], label='Val')
ax.set_xlabel('Epoch')
ax.set_ylabel('F1')
ax.set_title('F1 Score')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()

# %% Cell 4: Print best metrics
print("Best validation metrics:")
for k, v in results['best_metrics'].items():
    if k != 'confusion_matrix':
        print(f"  {k}: {v:.4f}")

print("\nConfusion matrix:")
cm = np.array(results['best_metrics']['confusion_matrix'])
print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

# %% Cell 5: Print hyperparameters
print("Hyperparameters used:")
for k, v in results['hyperparams'].items():
    print(f"  {k}: {v}")

# %% Cell 6: Load and inspect saved model
checkpoint = torch.load('psf_classifier.pt', map_location='cpu')
print("Saved model info:")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Val recall: {checkpoint['val_recall']:.4f}")
print(f"  Val F1: {checkpoint['val_f1']:.4f}")

# %% Cell 7: Test inference on a few samples
model = results['model']
model.eval()
device = results['device']

# Load some test data
with h5py.File('training_data.h5', 'r') as f:
    X_test = f['X'][:10]
    y_test = f['y'][:10]

X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

with torch.no_grad():
    logits = model(X_tensor)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()

print("\nSample predictions:")
for i in range(10):
    true_label = "BAD" if y_test[i] >= 0.5 else "GOOD"
    pred_label = "BAD" if probs[i] >= 0.5 else "GOOD"
    print(f"  Sample {i}: true={true_label}, pred={pred_label} (prob={probs[i]:.3f})")

# %% Cell 8: Visualize some predictions
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flat):
    img = X_test[i, 0]  # Remove channel dimension
    true_label = "BAD" if y_test[i] >= 0.5 else "GOOD"
    pred_label = "BAD" if probs[i] >= 0.5 else "GOOD"

    ax.imshow(img, cmap='Greys_r', origin='lower')
    color = 'green' if true_label == pred_label else 'red'
    ax.set_title(f"True: {true_label}\nPred: {pred_label} ({probs[i]:.2f})",
                 fontsize=9, color=color)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150)
plt.show()

# %% Cell 9: Load history from JSON (alternative way)
# Useful if you want to analyze results from a previous run
with open('psf_classifier_history.json', 'r') as f:
    saved_results = json.load(f)

print("Loaded from JSON:")
print(f"  Epochs trained: {len(saved_results['history'])}")
print(f"  Best recall: {saved_results['best_metrics']['recall']:.4f}")
