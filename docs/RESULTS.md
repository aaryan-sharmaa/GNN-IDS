# Final result sheet — GNN for Network Intrusion Detection

> **Experiment Status:** Section 11 complete and locked.
> Final test evaluation used frozen validation-selected thresholds. No further experiment runs are required.

---

## 1. Final Test Results

| Metric | GNN + Attention | GCN-only Ablation |
|---|---:|---:|
| **Test PR-AUC** | **0.784782** | **0.864075** |
| **Test ROC-AUC** | **0.903538** | **0.947848** |
| **Test F1** | **0.847998** | **0.879301** |
| Precision | 0.975941 | 0.865258 |
| Recall | 0.749713 | 0.893808 |
| Flagged flows | 4.146962% | 5.576443% |

### Frozen Model Selection

- **GNN + Attention:** Best validation PR-AUC at **epoch 11**
  - Frozen threshold: **0.51030290**
- **GCN-only:** Best validation PR-AUC at **epoch 1**
  - Frozen threshold: **0.49366927**
- Thresholds were selected using **validation data only**.
- Test data was not used for model or threshold selection.

---

## 2. Controlled Ablation Interpretation

This is **not a pure attention-only ablation**.

- **GNN + Attention:** graph structure + flow-level edge features + attention
- **GCN-only:** graph structure alone, using mean endpoint aggregation

Therefore, the experiment compares the complete flow-aware GNN+Attention architecture against a graph-structure-only GCN baseline.

---

## 3. Host-Pair Diagnostic — Validation Only

| Diagnostic | Result |
|---|---:|
| Validation flows | 354,814 |
| Unique unordered host pairs | 148 |
| Unique directed host pairs | 282 |
| Mixed-label unordered pairs | 35 / 148 |
| Flows in mixed-label pairs | 21,069 (5.938%) |
| Unordered-pair oracle max F1 | 0.905767 |
| GCN-only validation F1 | 0.882646 |
| GCN-only / F1 ceiling | **97.45%** |
| GCN-only distinct prediction scores | 92 |
| GCN-only max within-pair spread | ≤ 5.96 × 10⁻⁸ |
| GNN+Attention max within-pair spread | 0.0491 |

> **Note:** the GCN-only distinct-score count printed by both notebooks is **94**, not 92 (see
> *Known discrepancies* below).

### Key Diagnostic Findings

- GCN-only predictions are effectively constant for flows sharing the same **unordered host pair**; the tiny observed spread is numerical noise.
- 35 unordered validation pairs contain both benign and attack traffic, creating a structural ceiling for pair-level prediction.
- GNN+Attention can vary predictions within the same host pair through its flow-feature pathway.
- GCN-only reaches **97.45% of the unordered-pair oracle F1 ceiling** on validation.

---

## 4. Reproducibility

- Frozen thresholds unchanged — **PASS**
- Host-pair diagnostic accessed **no held-out/test data**
- No model, threshold, checkpoint, or final test result was modified.
- Diagnostic outputs saved under:

`outputs/final_experiment_backup/`

---

## 5. Known discrepancies between this sheet and the notebook outputs

This sheet was transcribed into notebook cell 192 by hand; two figures do not match what the
notebooks print. The values in the tables above are kept as originally locked; the notebook
outputs are recorded here so the difference is visible.

| Quantity | This sheet (locked) | Printed in both notebooks | Notebook location |
|---|---:|---:|---|
| GCN-only test ROC-AUC | 0.947848 | **0.947952** | original cell 226; clean cell 189 |
| GCN-only distinct validation scores | 92 | **94** | original cells 224 and 228; clean cells 187 and 191 |

* **ROC-AUC.** `GNN-IDS-FINAL-SUBMIT.ipynb` and `GNN-IDS-FINAL-SUBMIT-clean.ipynb` both print
  0.947952 for the GCN-only test ROC-AUC (test PR-AUC 0.864075, which agrees with this sheet).
  The 0.947848 in the table above is not reproduced by either notebook output. The cause has not
  been determined.
* **Distinct scores.** Both notebooks print 94 distinct GCN-only validation scores (against 148
  unordered host pairs). This sheet says 92. The cause has not been determined; no explanation
  is assumed here.
* Until reconciled, treat the notebook-printed values as the machine-produced figures and the
  sheet values as the figures that were locked in the write-up. A full-data re-run of
  `--stage controlled` writes the recomputed values to
  `outputs/final_experiment_backup/s11_final_test_results.csv` (`test_roc_auc`) and
  `s11_hostpair_diagnostic_summary.csv` (`groups` on the `ACTUAL frozen GCN-only` row).

---

# EXPERIMENT COMPLETE

**This result sheet is the locked numerical reference for the project report, PPT, and viva.**

> Source: `GNN-IDS-FINAL-SUBMIT-clean.ipynb`, cell 192. Figures were produced on a CUDA GPU; re-runs on other hardware may differ in the last digits (non-deterministic GPU scatter-add).
