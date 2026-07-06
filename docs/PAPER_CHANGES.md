# Exactly what to change in the manuscript

This maps each pipeline addition to a specific edit in the CISCom draft.
Numbers/tables below are placeholders from the synthetic smoke test — replace
with real CERT r4.2 numbers once you swap in `data_loader.py`.

## 1. Fix before anything else
- Author 7 affiliation/email placeholder (`<Name of the 7th Author's College>`,
  `author7@example.com`) — must be real or the author removed.
- Trim or delete §6 (Dashboard) — one paragraph with no screenshot doesn't
  carry its weight. If you keep it, add a screenshot and one sentence on how
  analysts actually used the fuzzy risk tiers (see #4) in it.

## 2. Abstract / Introduction — reframe the headline contribution
Lead with the leak-free rolling-origin protocol and the genuine soft-computing
pairing (GA + fuzzy layer), not the four drift metrics alone. Suggested new
abstract sentence:
> "Unlike prior CERT-based work relying on a single chronological split,
> Cyber DNA is evaluated under rolling-origin cross-validation with bootstrap
> confidence intervals, and its feature space is selected via a genetic
> algorithm rather than hand-picked blocks; a Mamdani fuzzy layer converts
> classifier output into analyst-facing Low/Medium/High risk tiers."

This directly answers the CISCom soft-computing scope question instead of
asserting XGBoost-as-soft-computing.

## 3. New §5.4 — Genetic-Algorithm Feature Selection
Insert after §5.3 (Behavioral Feature Taxonomy). Content: describe the
DEAP-based GA (binary chromosome over the 29-feature space, tournament
selection, two-point crossover, bit-flip mutation, fitness = mean F1 over
inner chronological folds *within the training partition only*, small
parsimony penalty). Cite this as the paper's evolutionary-computation
component. Source: `src/ga_feature_selection.py`.

Report, per outer fold: population size, generations, best fitness, and the
selected feature list (`results/summary_results.json:selected_features_last_fold`).

## 4. New §5.5 — Fuzzy Risk-Tiering Layer
Describe the Mamdani fuzzy system (`src/fuzzy_risk.py`) taking the XGBoost
probability plus BDS/IDP as inputs, with membership functions over
Low/Medium/High, feeding directly into the existing §7.5 SOC-deployment
discussion (replace "the 0.30 threshold would increase alerts from 16 to 37"
with the tier distribution table, e.g.
`results/summary_results.json:risk_tier_counts`).

## 5. Table 1a (new) — Rolling-origin fold definition
Replace the single train(1–52)/test(53–72) split description with a table of
all folds used (`src/evaluate.py::make_rolling_folds`), each row = fit weeks,
test weeks, malicious weeks in test.

## 6. Table 2 — add CI columns
Old:
| Model | Precision | Recall | F1 | AUPRC |
New:
| Model | Precision | Recall | F1 (mean ± std, k folds) | AUPRC (95% CI) |
Source: `summarize_folds()` for the mean/std row, `bootstrap_ci()` for the CI
on the most recent fold.

## 7. Table 4 — replace "unfair" literature comparison
Keep Tuor et al. / Yuan et al. numbers for context but add your own row:
"Sequence baseline (same chronological protocol)" using
`src/baseline_sequence.py` results (`seq_metrics` in `run_fold` output). This
directly answers the fairness objection instead of just noting it in prose.
NOTE: the sandbox here uses a lag-window MLP as the sequence baseline (no
GPU/PyTorch training time budget for a full LSTM in this environment, though
torch is installed and importable). Swap in a real `nn.LSTM` on your own
machine — same input tensors from `build_lag_windows()` work unchanged; see
docstring in `baseline_sequence.py`.

## 8. New §7.x — Statistical significance
Report the McNemar test vs. the 3σ heuristic (and vs. the sequence baseline)
instead of asserting superiority from the numbers alone.
Source: `mcnemar_test()`, `results/summary_results.json:mcnemar_vs_heuristic`.

## 9. Figure 2 replacement/addition — SHAP summary
Swap or supplement the gain-importance bar chart with `results/shap_summary.png`
(global SHAP summary plot) — standard practice now, and gain importance alone
is known to be biased toward high-cardinality/continuous features.

## 10. §7.5 Generalizability — one honest addition
Add a sentence noting the GA and fuzzy thresholds were tuned on this dataset's
class balance (0.48% positive rate) and would need re-tuning (not re-deriving
the whole pipeline) if the real-world positive rate differs — this pre-empts
the most likely reviewer question.

---

## Once you have the real CERT r4.2 CSVs
1. Upload the `r4.2/` release folder (`logon.csv`, `device.csv`, `email.csv`,
   `http.csv`, `psychometric.csv` if used, plus the answer-key insider
   scenario file).
2. In `main.py`, replace:
   `from data_gen import generate_cert_like_dataset, FEATURE_COLUMNS`
   with:
   `from data_loader import load_real_cert, FEATURE_COLUMNS`
   (schema is identical: `user_id, week, label` + the same raw feature names,
   so nothing downstream changes).
3. Re-run `python3 main.py` (full run, not `--quick`) — expect F1 to land back
   in the realistic ~40-55% range the original paper reports, since real
   insider scenarios are far subtler than the synthetic injected signal.
