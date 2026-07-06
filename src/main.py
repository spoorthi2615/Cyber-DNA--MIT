"""
End-to-end run of the strengthened Cyber DNA pipeline.

Pipeline: data -> leak-free DBS + drift metrics (per fold) -> GA feature
selection (per fold, train-only) -> Optuna-tuned XGBoost -> rolling-origin
chronological CV -> bootstrap CI -> McNemar tests vs. heuristic & sequence
baselines -> fuzzy risk tiers -> SHAP importance on the final fold.

Swap `generate_cert_like_dataset()` for your real-data loader (same schema:
user_id, week, label, + raw feature columns) once the actual CERT r4.2 (and
any extra fused source) files are available -- nothing else needs to change.
"""
import json
import numpy as np
import pandas as pd

import time
import argparse
import matplotlib.pyplot as plt

from data_loader import load_real_cert_r42
from data_gen import FEATURE_COLUMNS
from drift_metrics import build_dbs_features
from ga_feature_selection import run_ga_feature_selection
from quantum_feature_selection import run_quantum_feature_selection
from model_xgb import tune_xgb, fit_final_model, optimize_threshold
from evaluate import make_rolling_folds, compute_metrics, bootstrap_ci, mcnemar_test, summarize_folds
from baseline_sequence import build_lag_windows, fit_sequence_baseline
from fuzzy_risk import batch_risk_tiers

RAW_COLS = [c for c in FEATURE_COLUMNS if c not in ("BDS", "IDP", "BC", "SRC")]
RESULTS_DIR = "../results"


def heuristic_baseline(df, k_sigma=3.0):
    """3-sigma rolling heuristic on login_freq / usb_transfers, as in the paper's 7.2."""
    stats = df.groupby("user_id")[["login_freq", "usb_transfers"]].agg(["mean", "std"])
    stats.columns = ["_".join(c) for c in stats.columns]
    merged = df.merge(stats, on="user_id", how="left")
    thresh_login = merged["login_freq_mean"] + k_sigma * merged["login_freq_std"].fillna(0)
    thresh_usb = merged["usb_transfers_mean"] + k_sigma * merged["usb_transfers_std"].fillna(0)
    pred = ((merged["login_freq"] > thresh_login) | (merged["usb_transfers"] > thresh_usb)).astype(int)
    return pred.values


def run_fold(df, fit_weeks, test_weeks, ga_pop=24, ga_gen=10, optuna_trials=25, selector="ga"):
    train_cutoff = max(fit_weeks)
    all_weeks_needed = df["week"] <= max(test_weeks)
    fold_df = df[all_weeks_needed].copy()

    dbs_df = build_dbs_features(fold_df, RAW_COLS, train_week_cutoff=train_cutoff)
    feat_cols_full = RAW_COLS + ["BDS", "IDP", "BC", "SRC"]

    train_df = dbs_df[dbs_df["week"].isin(fit_weeks)]
    test_df = dbs_df[dbs_df["week"].isin(test_weeks)]

    t0 = time.time()
    if selector == "ga":
        selected_feats, best_fit, logbook = run_ga_feature_selection(
            train_df, feat_cols_full, pop_size=ga_pop, n_gen=ga_gen, n_folds=3, verbose=False
        )
    else:
        selected_feats, best_fit, logbook = run_quantum_feature_selection(
            train_df, feat_cols_full, pop_size=ga_pop, n_gen=ga_gen, n_folds=3, verbose=False
        )
    t1 = time.time()
    runtime = t1 - t0
    
    if len(selected_feats) < 3:
        selected_feats = feat_cols_full

    best_params, best_val = tune_xgb(train_df, selected_feats, n_trials=optuna_trials)
    clf = fit_final_model(train_df, selected_feats, best_params)

    # threshold tuned on a held-back slice of the training partition only (no test leakage)
    weeks_sorted = sorted(train_df["week"].unique())
    thresh_val_weeks = weeks_sorted[int(len(weeks_sorted) * 0.85):]
    thresh_val_df = train_df[train_df["week"].isin(thresh_val_weeks)]
    best_t, _ = optimize_threshold(clf, thresh_val_df, selected_feats)

    proba = clf.predict_proba(test_df[selected_feats])[:, 1]
    pred = (proba >= best_t).astype(int)
    metrics = compute_metrics(test_df["label"].values, pred, proba)

    # heuristic baseline on the same test slice
    heur_pred = heuristic_baseline(fold_df)[fold_df["week"].isin(test_weeks).values]

    # sequence baseline (lag-window MLP) fit on train, evaluated on test
    X_all, y_all, meta_all = build_lag_windows(dbs_df, selected_feats, window=3)
    train_mask = meta_all["week"].isin(fit_weeks).values
    test_mask = meta_all["week"].isin(test_weeks).values
    seq_metrics, seq_pred, seq_test_labels = None, None, None
    if train_mask.sum() > 20 and y_all[train_mask].sum() > 0 and test_mask.sum() > 5:
        seq_clf = fit_sequence_baseline(X_all[train_mask], y_all[train_mask], meta_all[train_mask])
        seq_proba = seq_clf.predict_proba(X_all[test_mask])[:, 1]
        seq_pred = (seq_proba >= 0.5).astype(int)
        seq_test_labels = y_all[test_mask]
        seq_metrics = compute_metrics(seq_test_labels, seq_pred, seq_proba)

    # fuzzy risk tiers on this fold's test predictions
    _, tiers = batch_risk_tiers(proba, test_df["BDS"].values, test_df["IDP"].values)

    return dict(
        metrics=metrics,
        heuristic_metrics=compute_metrics(test_df["label"].values, heur_pred, heur_pred.astype(float)),
        seq_metrics=seq_metrics,
        selected_feats=selected_feats,
        fitness=best_fit,
        logbook=logbook,
        runtime=runtime,
        best_params=best_params,
        threshold=best_t,
        y_true=test_df["label"].values,
        y_pred=pred,
        y_proba=proba,
        heur_pred=heur_pred,
        risk_tiers=tiers,
        clf=clf,
        test_feat_df=test_df[selected_feats],
    )


def plot_convergence(all_results):
    plt.figure(figsize=(10, 6))
    for sel, results in all_results.items():
        # Average the convergence history across folds
        all_histories = []
        for r in results:
            if sel == "ga":
                # DEAP logbook
                hist = [record["max"] for record in r["logbook"]]
            else:
                # QEA logbook
                hist = [record["max"] for record in r["logbook"]]
            all_histories.append(hist)
            
        mean_hist = np.mean(all_histories, axis=0)
        label = "GA (DEAP)" if sel == "ga" else "QEA (Quantum-Inspired)"
        plt.plot(range(1, len(mean_hist) + 1), mean_hist, marker='o', label=label, linewidth=2)
        
    plt.title("Feature Selection Convergence (Mean Fitness across Folds)")
    plt.xlabel("Generation")
    plt.ylabel("Mean F1 (minus penalty)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/convergence_overlay.png", dpi=300)
    print(f"Saved convergence plot to {RESULTS_DIR}/convergence_overlay.png")

def main(quick=False, selector_arg="ga"):
    print("Loading real dataset...")
    df = load_real_cert_r42(
        cert_dir="../data/cert_r4.2/r4.2", 
        ground_truth_path="../data/cert_r4.2/answers/insiders.csv"
    )

    n_folds = 2 if quick else 3
    folds = make_rolling_folds(df["week"].unique(), n_folds=n_folds, min_train_weeks=45, test_span=9)
    print(f"Running {len(folds)} rolling-origin folds:", folds)

    ga_pop, ga_gen, opt_trials = (10, 4, 8) if quick else (24, 15, 25)

    selectors = ["ga", "quantum"] if selector_arg == "both" else [selector_arg]
    all_results = {sel: [] for sel in selectors}

    for i, (fit_weeks, test_weeks) in enumerate(folds):
        print(f"\n--- Fold {i+1}: train weeks 1-{max(fit_weeks)}, test weeks {min(test_weeks)}-{max(test_weeks)} ---")
        for sel in selectors:
            print(f"Running selector: {sel.upper()}")
            res = run_fold(df, fit_weeks, test_weeks, ga_pop=ga_pop, ga_gen=ga_gen, optuna_trials=opt_trials, selector=sel)
            all_results[sel].append(res)
            print(f"[{sel.upper()}] Metrics:", res["metrics"])

    # Output summaries
    for sel in selectors:
        summary = summarize_folds([r["metrics"] for r in all_results[sel]])
        print(f"\n=== {sel.upper()} Rolling-origin CV summary ===")
        print(summary)
        
    if "ga" in selectors and "quantum" in selectors:
        # Print comparison table
        print("\n=== Feature Selection Comparison (Mean across folds) ===")
        print(f"{'Selector':<10} | {'# Features':<10} | {'Best Fitness':<12} | {'Runtime (s)':<12} | {'Downstream F1':<15} | {'Downstream AUPRC'}")
        print("-" * 80)
        
        convergence_export = {}
        
        for sel in selectors:
            mean_n_feat = np.mean([len(r["selected_feats"]) for r in all_results[sel]])
            mean_fit = np.mean([r["fitness"] for r in all_results[sel]])
            mean_rt = np.mean([r["runtime"] for r in all_results[sel]])
            
            f1s = [r["metrics"]["f1"] for r in all_results[sel]]
            auprcs = [r["metrics"]["auprc"] for r in all_results[sel]]
            mean_f1 = np.mean(f1s)
            std_f1 = np.std(f1s)
            mean_auprc = np.mean(auprcs)
            std_auprc = np.std(auprcs)
            
            f1_str = f"{mean_f1:.3f} ± {std_f1:.3f}"
            print(f"{sel.upper():<10} | {mean_n_feat:<10.1f} | {mean_fit:<12.4f} | {mean_rt:<12.1f} | {f1_str:<15} | {mean_auprc:.3f} ± {std_auprc:.3f}")

            # Collect convergence for export
            all_hist = []
            for r in all_results[sel]:
                all_hist.append([record["max"] for record in r["logbook"]])
            convergence_export[sel] = np.mean(all_hist, axis=0).tolist()

        plot_convergence(all_results)
        
        # Save detailed json including convergence
        with open(f"{RESULTS_DIR}/comparison_results.json", "w") as f:
            json.dump({
                "metrics": {sel: [r["metrics"] for r in all_results[sel]] for sel in selectors},
                "convergence": convergence_export
            }, f, indent=2)
            
        # SHAP calculation for the last fold of QEA (or GA if QEA missing)
        try:
            import shap
            last_qea = all_results["quantum"][-1]
            explainer = shap.TreeExplainer(last_qea["clf"])
            # Subsample for speed
            X_test_shap = last_qea["test_feat_df"].sample(n=min(5000, len(last_qea["test_feat_df"])), random_state=42)
            shap_values = explainer.shap_values(X_test_shap)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            shap_df = pd.DataFrame({
                "feature": last_qea["selected_feats"],
                "mean_abs_shap": mean_abs_shap
            }).sort_values(by="mean_abs_shap", ascending=False)
            
            shap_export = shap_df.head(15).to_dict(orient="records")
            with open(f"{RESULTS_DIR}/shap_summary.json", "w") as f:
                json.dump(shap_export, f, indent=2)
            print(f"\nExtracted top 15 SHAP features to {RESULTS_DIR}/shap_summary.json")
        except ImportError:
            print("\nSHAP library not found, skipping feature importance extraction.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--selector", choices=["ga", "quantum", "both"], default="ga")
    args = parser.parse_args()
    main(quick=args.quick, selector_arg=args.selector)
