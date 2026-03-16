#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

root = Path(__file__).resolve().parents[1]
raw_dir = root / "demo_data" / "tabular" / "raw"
fdb_path = raw_dir / "FDB.csv"
ldb_path = raw_dir / "LDB.csv"

if not fdb_path.exists() or not ldb_path.exists():
    raise SystemExit(f"Missing raw files. Expected {fdb_path} and {ldb_path}")

fdb = pd.read_csv(fdb_path, sep=";")
ldb = pd.read_csv(ldb_path, sep=";")

key = "Key" if "Key" in fdb.columns else fdb.columns[0]
label_col = "Low-High" if "Low-High" in ldb.columns else [c for c in ldb.columns if c != key][0]

merged = fdb.merge(ldb[[key, label_col]], on=key, how="inner")
X = merged.drop(columns=[label_col, key])
y = merged[label_col]

Xn = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
feature_cols = list(Xn.columns)
fcols = [f"f{i}" for i in range(len(feature_cols))]

tab_dir = root / "demo_data" / "tabular"
tab_dir.mkdir(parents=True, exist_ok=True)

train_df = Xn.copy()
train_df.columns = fcols
train_df["label"] = y.values
train_df.to_csv(tab_dir / "real_train.csv", index=False)

infer_df = Xn.copy()
infer_df.columns = fcols
infer_df.to_csv(tab_dir / "real_infer.csv", index=False)

pd.DataFrame({"f_col": fcols, "original_feature": feature_cols}).to_csv(tab_dir / "real_feature_map.csv", index=False)

print(f"Wrote {tab_dir/'real_train.csv'} rows={train_df.shape[0]} features={Xn.shape[1]}")
print(f"Wrote {tab_dir/'real_infer.csv'} rows={infer_df.shape[0]} features={Xn.shape[1]}")
