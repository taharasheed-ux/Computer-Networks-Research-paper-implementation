"""Quick script to train only MLP and Ensemble using existing baseline results."""
import json
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import models
try:
    from models import fit_predict_mlp, fit_predict_voting
except ImportError:
    from src.models import fit_predict_mlp, fit_predict_voting

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def _metrics(y_true, y_pred):
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )

# Load data
data = np.load("data/sfe_offline.npz")
X = data["X"]
y = data["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)

# Load existing results
os.makedirs("results_fri_gpu", exist_ok=True)
results_file = Path("results_fri_gpu/metrics.json")
if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)
    print("Loaded existing results:", list(results.keys()))
else:
    results = {}

# Train MLP
print("\n[1/2] Training Deep Learning (MLP - PyTorch)...")
import torch
try:
    from mlp_torch import TorchSklearnWrapper
except ImportError:
    from src.mlp_torch import TorchSklearnWrapper

y_pred = fit_predict_mlp(X_train, X_test, y_train)
results["MLP (Deep Learning)"] = _metrics(y_test, y_pred)
print(f"  ✓ MLP - Acc: {results['MLP (Deep Learning)']['accuracy']:.4f}, F1: {results['MLP (Deep Learning)']['f1']:.4f}")

# Save MLP model
mlp_wrapper = TorchSklearnWrapper(epochs=20)
mlp_wrapper.fit(X_train, y_train)
torch.save(mlp_wrapper.model.state_dict(), Path("results_fri_gpu", "mlp_model.pth"))
print("  Saved MLP model to mlp_model.pth")

with open(results_file, "w", encoding="utf-8") as fp:
    json.dump(results, fp, indent=2)

# Train Ensemble
print("\n[2/2] Training Ensemble (Voting: RF + MLP)...")
try:
    y_pred = fit_predict_voting(X_train, X_test, y_train)
    results["Ensemble (Voting)"] = _metrics(y_test, y_pred)
    print(f"  ✓ Ensemble - Acc: {results['Ensemble (Voting)']['accuracy']:.4f}, F1: {results['Ensemble (Voting)']['f1']:.4f}")
except Exception as e:
    print(f"  ✗ Ensemble training failed: {e}")
    print("  Skipping ensemble...")

# Final save
with open(results_file, "w", encoding="utf-8") as fp:
    json.dump(results, fp, indent=2)

print("\n✓ Done! Results saved to results_fri_gpu/metrics.json")
print("\nFinal Results:")
for name, m in results.items():
    print(f"{name:20s} Acc {m['accuracy']:.4f}  F1 {m['f1']:.4f}")
