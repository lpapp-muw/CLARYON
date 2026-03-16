#!/usr/bin/env python3
"""Patch pipeline.py to:
1. Import quantum and CNN model modules (so @register fires)
2. Amplitude-encode data before quantum models
3. Use encoded data for predict_proba

Run from project root:
    cd ~/claryon
    source .venv/bin/activate
    python scripts/patch_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

PIPELINE = Path(__file__).resolve().parent.parent / "claryon" / "pipeline.py"

content = PIPELINE.read_text()
changes = 0

# ── Fix 1: Add quantum + CNN to model imports ──
OLD_IMPORTS = '''    optional = [
        "claryon.models.classical.xgboost_",
        "claryon.models.classical.lightgbm_",
        "claryon.models.classical.catboost_",
        "claryon.models.classical.tabpfn_",
        "claryon.models.classical.debinn_",
        "claryon.models.classical.tabm_",
        "claryon.models.classical.realmlp_",
        "claryon.models.classical.modernnca_",
    ]'''

NEW_IMPORTS = '''    optional = [
        "claryon.models.classical.xgboost_",
        "claryon.models.classical.lightgbm_",
        "claryon.models.classical.catboost_",
        "claryon.models.classical.tabpfn_",
        "claryon.models.classical.debinn_",
        "claryon.models.classical.tabm_",
        "claryon.models.classical.realmlp_",
        "claryon.models.classical.modernnca_",
        "claryon.models.classical.cnn_2d",
        "claryon.models.classical.cnn_3d",
        "claryon.models.quantum.kernel_svm",
        "claryon.models.quantum.qcnn_muw",
        "claryon.models.quantum.qcnn_alt",
        "claryon.models.quantum.vqc",
        "claryon.models.quantum.hybrid",
    ]'''

if "claryon.models.quantum.kernel_svm" not in content:
    content = content.replace(OLD_IMPORTS, NEW_IMPORTS)
    changes += 1
    print("[1/3] Added quantum + CNN model imports")
else:
    print("[1/3] Quantum imports already present — skipping")


# ── Fix 2: Amplitude-encode for quantum models before fit ──
OLD_TRAIN = '''                try:
                    model = model_cls(**model_entry.params)
                    model.fit(X_train, y_train, ds.task_type)'''

NEW_TRAIN = '''                try:
                    model = model_cls(**model_entry.params)

                    # Amplitude-encode for quantum models
                    X_tr_use, X_te_use = X_train, X_test
                    if model_entry.type == "tabular_quantum":
                        from .encoding.amplitude import amplitude_encode_matrix
                        X_tr_use, enc_info = amplitude_encode_matrix(X_train)
                        X_te_use, _ = amplitude_encode_matrix(X_test, pad_len=enc_info.encoded_dim)
                        logger.info("  Amplitude encoded: %d features -> %d (qubits=%d)",
                                    X_train.shape[1], enc_info.encoded_dim, enc_info.n_qubits)

                    model.fit(X_tr_use, y_train, ds.task_type)'''

if "tabular_quantum" not in content:
    content = content.replace(OLD_TRAIN, NEW_TRAIN)
    changes += 1
    print("[2/3] Added amplitude encoding for quantum models")
else:
    print("[2/3] Quantum encoding already present — skipping")


# ── Fix 3: Use encoded data for predict ──
OLD_PREDICT = '''                    if ds.task_type == TaskType.REGRESSION:
                        predicted = model.predict(X_test)
                        probs = None
                    else:
                        probs = model.predict_proba(X_test)
                        predicted = np.argmax(probs, axis=1)'''

NEW_PREDICT = '''                    if ds.task_type == TaskType.REGRESSION:
                        predicted = model.predict(X_te_use)
                        probs = None
                    else:
                        probs = model.predict_proba(X_te_use)
                        predicted = np.argmax(probs, axis=1)'''

if "X_te_use" not in content:
    content = content.replace(OLD_PREDICT, NEW_PREDICT)
    changes += 1
    print("[3/3] Fixed predict to use encoded data")
else:
    print("[3/3] Predict already uses encoded data — skipping")

PIPELINE.write_text(content)
print(f"\nDone. {changes} patches applied to {PIPELINE}")
