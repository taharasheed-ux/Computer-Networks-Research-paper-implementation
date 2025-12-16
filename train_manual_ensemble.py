"""Manual ensemble implementation - combines RF and MLP predictions."""
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from mlp_torch import TorchSklearnWrapper
except ImportError:
    from src.mlp_torch import TorchSklearnWrapper

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
results_file = Path("results_fri_gpu/metrics.json")
with open(results_file) as f:
    results = json.load(f)

print("\n[Manual Ensemble] Training RF + MLP with probability averaging...")

# Train RF
print("  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
print("  ✓ RF trained")

# Train MLP  
print("  Training MLP (GPU)...")
mlp = TorchSklearnWrapper(epochs=20)
mlp.fit(X_train, y_train)
mlp_probs = mlp.predict_proba(X_test)[:, 1]
print("  ✓ MLP trained")

# Average probabilities
ensemble_probs = (rf_probs + mlp_probs) / 2.0
ensemble_preds = (ensemble_probs > 0.5).astype(int)

results["Ensemble (Manual)"] = _metrics(y_test, ensemble_preds)
print(f"\n✓ Ensemble - Acc: {results['Ensemble (Manual)']['accuracy']:.4f}, F1: {results['Ensemble (Manual)']['f1']:.4f}")

# Save
with open(results_file, "w", encoding="utf-8") as fp:
    json.dump(results, fp, indent=2)

print("\n✓ Done! Updated results in results_fri_gpu/metrics.json")
print("\nAll Results:")
for name, m in results.items():
    print(f"{name:25s} Acc {m['accuracy']:.4f}  Prec {m['precision']:.4f}  Rec {m['recall']:.4f}  F1 {m['f1']:.4f}")
