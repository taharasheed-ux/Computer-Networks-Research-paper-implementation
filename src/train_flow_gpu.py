# src/train_flow_gpu.py
"""Command-line script to run baseline classifiers and GPU S_FE on the
CIC-IDS flow CSV.

Example:
    python train_flow_gpu.py --csv data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv \
        --output results/
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

try:
    from features import load_flow_csv, select_numeric_features
except ImportError:  # when executed as -m src.train_flow_gpu
    from src.features import load_flow_csv, select_numeric_features
try:
    from models import (
        fit_predict_lr,
        fit_predict_lda,
        fit_predict_pca_lr,
        fit_predict_rf,
        fit_predict_mlp,
        fit_predict_voting,
    )
except ImportError:
    from src.models import (
        fit_predict_lr,
        fit_predict_lda,
        fit_predict_pca_lr,
        fit_predict_rf,
        fit_predict_mlp,
        fit_predict_voting,
    )
try:
    from sfe_gpu import s_fe_stochastic_search
except ImportError:
    from src.sfe_gpu import s_fe_stochastic_search


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def _metrics(y_true, y_pred) -> Dict[str, float]:
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def main():
    parser = argparse.ArgumentParser(description="GPU Flow-based NIDS Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", help="Path to CIC flow CSV file (flow-stat baseline)")
    group.add_argument("--payload_npz", help="Path to pre-computed S_FE payload dataset (.npz)")

    parser.add_argument("--output", default="results", help="Directory to save CSV/plots")
    parser.add_argument("--sfe_iter", type=int, default=30)
    parser.add_argument("--sfe_restart", type=int, default=1)
    args = parser.parse_args()

    if args.payload_npz:
        import numpy as _np  # local import to avoid confusion
        data = _np.load(args.payload_npz)
        X = data["X"]
        y = data["y"]
        feat_names = [s for s in data["feature_names"]]
    else:
        df = load_flow_csv(args.csv)
        X, y, feat_names = select_numeric_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    results = {}

    # Logistic Regression
    print("\n[1/6] Training Logistic Regression...")
    y_pred = fit_predict_lr(X_train, X_test, y_train)
    results["LR"] = _metrics(y_test, y_pred)
    print(f"  ✓ LR - Acc: {results['LR']['accuracy']:.4f}, F1: {results['LR']['f1']:.4f}")
    # Save incrementally
    with open(Path(args.output, "metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    # LDA
    print("\n[2/6] Training Linear Discriminant Analysis...")
    y_pred = fit_predict_lda(X_train, X_test, y_train)
    results["LDA"] = _metrics(y_test, y_pred)
    print(f"  ✓ LDA - Acc: {results['LDA']['accuracy']:.4f}, F1: {results['LDA']['f1']:.4f}")
    with open(Path(args.output, "metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    # PCA baseline – only when we have enough features
    if len(feat_names) > 10:
        y_pred = fit_predict_pca_lr(X_train, X_test, y_train, n_components=10)
        results["PCA(10)+LR"] = _metrics(y_test, y_pred)
        with open(Path(args.output, "metrics.json"), "w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2)

    # Random Forest (paper’s main classifier)
    print("\n[3/6] Training Random Forest...")
    y_pred = fit_predict_rf(X_train, X_test, y_train)
    results["RandomForest"] = _metrics(y_test, y_pred)
    print(f"  ✓ Random Forest - Acc: {results['RandomForest']['accuracy']:.4f}, F1: {results['RandomForest']['f1']:.4f}")
    with open(Path(args.output, "metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    # Deep Learning (MLP)
    print("\n[4/6] Training Deep Learning (MLP - PyTorch)...")
    import torch
    
    # Train and get predictions
    y_pred = fit_predict_mlp(X_train, X_test, y_train)
    results["MLP (Deep Learning)"] = _metrics(y_test, y_pred)
    print(f"  ✓ MLP - Acc: {results['MLP (Deep Learning)']['accuracy']:.4f}, F1: {results['MLP (Deep Learning)']['f1']:.4f}")
    
    # Save MLP model
    mlp_wrapper = TorchSklearnWrapper(epochs=20)
    mlp_wrapper.fit(X_train, y_train)
    torch.save(mlp_wrapper.model.state_dict(), Path(args.output, "mlp_model.pth"))
    print("  Saved MLP model to mlp_model.pth")
    
    with open(Path(args.output, "metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    # Ensemble (Voting: RF + MLP)
    print("\n[5/6] Training Ensemble (Voting: RF + MLP)...")
    try:
        y_pred = fit_predict_voting(X_train, X_test, y_train)
        results["Ensemble (Voting)"] = _metrics(y_test, y_pred)
        print(f"  ✓ Ensemble - Acc: {results['Ensemble (Voting)']['accuracy']:.4f}, F1: {results['Ensemble (Voting)']['f1']:.4f}")
    except Exception as e:
        print(f"  ✗ Ensemble training failed: {e}")
        print("  Skipping ensemble and continuing...")
    
    with open(Path(args.output, "metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    # S_FE GPU search (only makes sense if >1 feature)
    if len(feat_names) > 5:  # skip when already the engineered 5 features
        print("\n[6/6] Running S_FE Feature Selection...")
        # Ensure top_k doesn't exceed available features
        top_k = min(20, len(feat_names))
        sel_features, best_val = s_fe_stochastic_search(
            X, y, feat_names, top_k=top_k, n_iter=args.sfe_iter, n_restarts=args.sfe_restart
        )
        sel_idx = [feat_names.index(f) for f in sel_features]
        X_sel = X[:, sel_idx]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sel, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
        )
        y_pred = fit_predict_lr(X_tr, X_te, y_tr)
        results["SFE_selected_LR"] = _metrics(y_te, y_pred)
        print(f"  ✓ S_FE - Selected {len(sel_features)} features, F1: {results['SFE_selected_LR']['f1']:.4f}")

    # persist
    # persist
    os.makedirs(args.output, exist_ok=True)
    with open(Path(args.output, "metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    # only write S_FE details if we actually ran the search
    if len(feat_names) > 5:
        with open(Path(args.output, "sfe_selected_features.json"), "w", encoding="utf-8") as fp:
            json.dump(dict(selected=sel_features, best_val=best_val), fp, indent=2)

    print("Saved results →", args.output)
    for name, m in results.items():
        print(f"{name:15s} Acc {m['accuracy']:.4f}  Prec {m['precision']:.4f}  "
              f"Rec {m['recall']:.4f}  F1 {m['f1']:.4f}")


if __name__ == "__main__":
    main()
