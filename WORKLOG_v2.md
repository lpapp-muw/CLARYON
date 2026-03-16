# CLARYON — Work Log

**Purpose**: Cross-session continuity document. Read this FIRST at the start of every new chat.

---

## 1. Current State

| Field | Value |
|---|---|
| **Phase** | 9 — new quantum methods + config fixes |
| **Last completed item** | Pipeline wiring complete (v0.8.0-wired), structured methods.tex installed |
| **Next item to build** | Port 5 quantum methods from Moradi papers, add GDQ score, fix NIfTI patterns, update README, cleanup scripts |
| **Blockers** | None |
| **CURRENT_PHASE** | 9 |
| **CURRENT_STEP** | quantum_methods_port |

---

## 2. Session Log

### Sessions 1-4 — 2026-03-16

See previous WORKLOG versions. Summary:
- Phases 0-7 built autonomously by Claude Code (~50 min)
- Pipeline wiring (stages 5-7) done in session 5 (~12 min)
- Manual fixes: __main__.py, quantum/CNN imports, amplitude encoding in pipeline
- Structured methods.tex generator installed (method_descriptions.yaml text registry)
- Full pipeline verified: iris classical + quantum + NIfTI CNN

### Current commit history
```
v0.8.0-wired — all pipeline stages wired
70ad976 — fix: __main__.py, pipeline quantum/CNN imports, amplitude encoding, demo configs
3514267 — demo: iris classical + quantum + configs verified
```

---

## 3. Hard Facts & Lessons Learned

### HF-001 through HF-013: See previous sessions.

### HF-014: Moradi quantum methods use Mottonen state preparation
**Fact**: All quantum methods from Moradi et al. (2022, 2023) use `qml.templates.MottonenStatePreparation` for amplitude encoding, NOT `qml.AmplitudeEmbedding`. Both achieve the same log2(N) encoding but Mottonen uses a specific decomposition. The existing CLARYON kernel_svm uses AmplitudeEmbedding. Both are valid for simulator-only use. New methods should use MottonenStatePreparation to match the published papers; existing methods keep AmplitudeEmbedding.

### HF-015: sqKSVM is actually a simplified GP, not SVM
**Fact**: The "simplified quantum kernel SVM" from Moradi et al. 2022 computes μ = y_train @ K(X_train, X_test), which is a linear kernel prediction without the SVC wrapper. It's closer to a kernel GP without noise. Keep the published name for citation accuracy but document this.

### HF-016: Quantum distance classifier uses class-separated training
**Fact**: The qDC methods split training data by class, compute similarity between each class's samples and each test sample, and predict based on which class has the maximum similarity. This is a nearest-class-mean style classifier in quantum kernel space. Both Hadamard and SWAP variants use this pattern.

### HF-017: SWAP test uses 2n+1 qubits
**Fact**: The SWAP-based distance classifier needs two registers (one per state) plus an ancilla qubit: 2*log2(features) + 1 qubits. For 8 features: 2*3+1 = 7 qubits. The Hadamard variant needs only n+1: log2(features) + 1.

### HF-018: qNN uses PyTorch interface with per-class circuits
**Fact**: The quantum neural network from Moradi et al. creates one quantum circuit PER CLASS (one-vs-rest style), each with its own trainable weights. Uses PyTorch interface (not autograd). Margin-based loss. This is different from the VQC stub already in CLARYON.

### HF-019: NIfTI image_pattern hardcoded to PET
**Fact**: The NIfTI loader defaults to `*PET*.nii*` for image files. This is nuclear medicine specific. The config should allow `image_pattern` (user-defined glob), defaulting to `*` (match all non-mask NIfTI files).

### HF-020: QCNN MUW paper under revision
**Fact**: The QCNN MUW method paper is currently under revision. Add a placeholder citation in method_descriptions.yaml that says "Papp et al., under revision". Update when published.

---

## 4. TODO for Next Session

### Task 1: Port 5 quantum models from Moradi papers (source in source_archive/)

| Model | Source | Target | Key implementation notes |
|---|---|---|---|
| sq_kernel_svm | `moradi_qml/sqKSVM.py` | `claryon/models/quantum/sq_kernel_svm.py` | Mottonen + inverse Mottonen + projector kernel → linear prediction μ = y @ K(train, test). No SVC wrapper. |
| qdc_hadamard | `moradi_qml/qDC_Hadamard_Test.py` + `moradi_em/qDS.py` | `claryon/models/quantum/qdc_hadamard.py` | Ancilla + controlled Mottonen + PauliX flip + Hadamard → PauliZ. Class-separated: max similarity per class → predict. n+1 qubits. |
| qdc_swap | `moradi_qml/qDC_Swap_Test.py` | `claryon/models/quantum/qdc_swap.py` | Two registers + CSWAP per qubit pair + ancilla. Same class-separated prediction. 2n+1 qubits. |
| quantum_gp | `moradi_em/GP.py` | `claryon/models/quantum/quantum_gp.py` | Full GP: K(train,train) + noise, K(train,test), posterior mean+cov, sigmoid for classification. Mottonen kernel. |
| qnn | `moradi_em/qnn_1.py` | `claryon/models/quantum/qnn.py` | REPLACE existing VQC stub. Per-class circuits with Rot+CNOT layers. PyTorch training with margin loss. |

### Task 2: Add Geometric Difference score
- Source: `moradi_qml/g_score.py`
- Target: `claryon/evaluation/geometric_difference.py`
- GDQ = √(spectral_norm(K_Q^½ @ K_C^-1 @ K_Q^½))
- Register as evaluation utility (not a model)
- Add to method_descriptions.yaml

### Task 3: Fix NIfTI image_pattern
- Add `image_pattern` field to `ImagingDataConfig` in `config_schema.py` (default: `"*"`)
- Update `claryon/io/nifti.py` to use `image_pattern` instead of hardcoded `*PET*`
- Pipeline passes `image_pattern` from config to loader
- Update demo configs

### Task 4: Update README.md
- Author: Laszlo Papp, PhD, EANM AI Committee member, laszlo.papp@meduniwien.ac.at
- Add references to Moradi et al. 2022 and 2023 papers
- Add QCNN MUW citation placeholder (under revision)
- Remove OCT/biophotonics as co-developed domains (it's an intended use, not an origin)

### Task 5: Update method_descriptions.yaml
- Add text blocks for all 5 new quantum models
- Add GDQ score description
- Add QCNN MUW citation placeholder: "Papp et al., under revision"
- Ensure all \cite keys match

### Task 6: Cleanup scripts/
- DELETE: patch_pipeline.py, install_structured_methods.py
- MOVE: method_descriptions.yaml → already in claryon/reporting/
- KEEP: setup_experiments.py, run_validation.sh

### Task 7: Create Jupyter notebooks for main modules
- 8 notebooks in examples/notebooks/ covering: quickstart, tabular, quantum models, NIfTI imaging, explainability, results dashboard, radiomics, custom model guide
- All self-contained, using demo data, runnable without external setup

---

## 5. References

- Moradi S, ..., Papp L. "Clinical data classification with noisy intermediate scale quantum computers." Sci Rep 12, 1851 (2022). https://doi.org/10.1038/s41598-022-05971-9
- Moradi S, ..., Papp L. "Error mitigation enables PET radiomic cancer characterization on quantum computers." Eur J Nucl Med Mol Imaging 50, 3826-3837 (2023). https://doi.org/10.1007/s00259-023-06362-6
- Papp L, et al. "Quantum Convolutional Neural Networks for Predicting ISUP Grade risk in [68Ga]Ga-PSMA Primary Prostate Cancer Patients." Under revision.
