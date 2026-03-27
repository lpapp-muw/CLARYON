#!/bin/bash
cd ~/claryon
source .venv/bin/activate
python3 << 'PYEOF'
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score

datasets = ["wisconsin", "cervical", "hcc", "psma11"]

for ds in datasets:
    base = Path(f"Results/eanm_abstract/{ds}")
    if not base.exists():
        print(f"{ds}: SKIPPED (no results)")
        continue

    print(f"\n=== {ds} ===")
    rows = []
    
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        model_name = model_dir.name
        if model_name in ("config_used.yaml", "run_info.json"):
            continue
        
        pred_files = sorted(model_dir.glob("seed_*/fold_*/Predictions.csv"))
        if not pred_files:
            continue
        
        baccs, aucs, sens, specs = [], [], [], []
        for pf in pred_files:
            df = pd.read_csv(pf, sep=";")
            y_true = df["Actual"].values
            y_pred = df["Predicted"].values
            p1 = df["P1"].values
            
            baccs.append(balanced_accuracy_score(y_true, y_pred))
            try:
                aucs.append(roc_auc_score(y_true, p1))
            except:
                pass
            sens.append(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
            specs.append(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
        
        row = {
            "model": model_name,
            "bacc": round(np.mean(baccs), 3),
            "bacc_std": round(np.std(baccs), 3),
            "auc": round(np.mean(aucs), 3) if aucs else float("nan"),
            "auc_std": round(np.std(aucs), 3) if aucs else float("nan"),
            "sensitivity": round(np.mean(sens), 3),
            "sensitivity_std": round(np.std(sens), 3),
            "specificity": round(np.mean(specs), 3),
            "specificity_std": round(np.std(specs), 3),
        }
        rows.append(row)
        print(f"  {model_name:20s} BACC={row['bacc']:.3f}±{row['bacc_std']:.3f}  AUC={row['auc']:.3f}±{row['auc_std']:.3f}")
    
    if rows:
        summary = pd.DataFrame(rows)
        out = base / "metrics_summary.csv"
        summary.to_csv(out, sep=";", index=False)
        print(f"  -> Wrote {out}")

print("\nDone.")
PYEOF
