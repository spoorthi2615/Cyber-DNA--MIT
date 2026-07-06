# Cyber DNA v2 — Strengthened Pipeline for CISCom 2026

This is a rebuilt version of the Cyber DNA insider-threat pipeline, designed
to close the specific gaps identified in review of the original paper draft,
and to give the paper a genuine soft-computing contribution (not just an
asserted one) for CISCom's scope.

## What's new vs. the original paper, and why

| Addition | File | Fixes |
|---|---|---|
| Rolling-origin chronological CV (multiple expanding-window folds instead of one train/test split) | `evaluate.py::make_rolling_folds` | Single-split variance concern — with only ~322 malicious weeks, one split isn't enough to trust a point estimate |
| Bootstrap 95% CIs on F1/AUPRC | `evaluate.py::bootstrap_ci` | No uncertainty quantification in the original tables |
| McNemar significance test vs. heuristic & sequence baselines | `evaluate.py::mcnemar_test` | "48% beats 2%" was asserted, never statistically tested |
| **Genetic-algorithm feature selection** (DEAP), fitness evaluated via leak-free inner chronological CV | `ga_feature_selection.py` | Original Table 3 hand-picks feature *blocks* one at a time — this is manual ablation, not selection. GA searches the real subset space and gives the paper a legitimate evolutionary-computation / soft-computing contribution |
| **Fuzzy (Mamdani) risk-tiering layer** on top of XGBoost's probability + BDS/IDP | `fuzzy_risk.py` | Turns a raw probability into an interpretable Low/Medium/High tier for SOC analysts — a second real soft-computing component, and a direct extension of the paper's existing "SOC deployment" discussion (§7.5) |
| Optuna (TPE/Bayesian) hyperparameter tuning, done inside the training partition only | `model_xgb.py` | Matches what reference [21] does, wired correctly into the chronological protocol |
| **Fair sequence baseline**, evaluated under the *same* chronological protocol | `baseline_sequence.py` | Original Table 4 compares against *other papers'* LSTM numbers under random splits — an admittedly unfair comparison. This answers "what does a sequence model get under YOUR protocol?" directly. (Uses lag-window + MLP as a stand-in for a full LSTM — this sandbox can't install PyTorch; see the docstring in that file for the exact drop-in swap to a real `nn.LSTM` once you're on your own machine.) |
| SHAP explainability (global summary + local waterfall) | `shap_explain.py` | Original Fig. 2 only shows raw gain importance, which is biased and gives no per-prediction explanation |
| Leak-free normalization + BDS/IDP/BC/SRC recomputed **per fold** | `drift_metrics.py` | Closes a subtle leakage gap: with multiple folds, T_base and min/max must be refit per fold, not once globally |
| Real-data loader with identical output schema to the synthetic generator | `data_loader.py` | Lets you swap synthetic → real CERT r4.2 (+ extra fused sources) by changing one import line |

## Directory layout
```
cyberdna/
  src/
    data_gen.py           synthetic CERT-r4.2-shaped data (works today, no upload needed)
    data_loader.py         REAL CERT r4.2 loader — point at your CSVs (see below)
    drift_metrics.py       leak-free normalization + BDS/IDP/BC/SRC
    ga_feature_selection.py  genetic algorithm feature selection (DEAP)
    fuzzy_risk.py           fuzzy Mamdani risk-tiering layer
    model_xgb.py            Optuna-tuned XGBoost
    baseline_sequence.py    fair lag-window sequence baseline
    evaluate.py             rolling CV, bootstrap CI, McNemar test
    shap_explain.py         SHAP global/local explanations
    main.py                 orchestrates the full pipeline
  results/                  JSON summaries + SHAP plots land here
  data/                     synthetic dataset cache
```

## Running it today (synthetic data, validates the whole pipeline)
```bash
cd src
python3 main.py --quick      # ~1-2 min, small pop/gen for a fast smoke test
python3 main.py              # full-size synthetic run (1000 users, larger GA/Optuna budget)
```
Note: on synthetic data the injected malicious signal is deliberately easy to
separate (to prove every stage runs correctly), so you'll see F1 near 1.0.
That is **expected and not the number to report** — it's a plumbing test, not
a result. Real CERT r4.2 will land back in the realistic ~40-55% F1 range
the paper reports, because the real insider scenarios are subtle.

## Switching to real CERT r4.2 data
1. Upload the CERT r4.2 release folder (needs `logon.csv`, `device.csv`,
   `email.csv`, `http.csv`, and the ground-truth answer key identifying
   which user/date-ranges are the injected insider scenarios).
2. In `main.py`, change:
   ```python
   from data_gen import generate_cert_like_dataset
   ```
   to:
   ```python
   from data_loader import load_real_cert_r42
   ```
   and call it with your paths:
   ```python
   df = load_real_cert_r42(cert_dir="/mnt/user-data/uploads/cert_r42",
                            ground_truth_path="/mnt/user-data/uploads/cert_r42/answers/insiders.csv")
   ```
3. Some raw features in `data_loader.py` are left as placeholders (0) because
   they need extra joins your specific r4.2 extract may or may not include
   (e.g., `email_recv_count` needs a reverse join on the `to` field, `new_pc_count`
   needs per-user PC history). These are clearly marked with comments — fill
   them in once you see your actual file structure, since CERT redistributions
   vary slightly in columns.

## Adding an extra dataset (you mentioned this might strengthen the paper)
`data_gen.py::generate_cert_like_dataset(extra_source=...)` and
`data_loader.py::load_real_cert_r42(extra_files=...)` both support fusing in
an additional per-user modality — e.g., badge/WiFi geolocation logs (movement
entropy) or static OCEAN psychometric scores, both of which have precedent in
the insider-threat literature as complementary signals. If you tell me which
extra dataset you're planning to add, I can wire up the specific
feature-extraction logic for it.

## What still needs YOUR input to finish
- The actual CERT r4.2 files (I don't have them / can't download them here —
  the network in this sandbox is restricted to package registries).
- Confirmation of your ground-truth answer-key format (column names vary
  slightly between r4.2 redistributions).
- A decision on the extra dataset, if you're adding one.
- Full-scale run (not `--quick`) once real data is in, with GA/Optuna budgets
  turned up — this will take longer than the synthetic smoke test.

## Paper-writing changes that pair with this code
- Report Table 2/3 numbers as **mean ± std across rolling-origin folds**,
  not a single split.
- Add the bootstrap CI columns to those tables.
- Add a row to Table 4 for "Sequence baseline (same protocol)" using the
  `baseline_sequence.py` numbers, next to the existing (protocol-mismatched)
  literature comparisons — and say so explicitly.
- Add a new subsection "5.4 Evolutionary Feature Selection" describing the
  GA in place of / alongside the current manual block-ablation Table 3.
- Add a new subsection "6.x Fuzzy Risk Tiering" extending the dashboard/SOC
  discussion, with the tier-distribution table as a figure.
- Fix the placeholder 7th-author affiliation before submission.
