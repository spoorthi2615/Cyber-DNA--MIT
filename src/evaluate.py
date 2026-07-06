"""
Rolling-origin chronological CV + statistical rigor layer.

The original paper reports ONE train(1-52)/test(53-72) split. With only
~322 malicious weeks total, a single split has real variance. This module
runs SEVERAL chronological folds (each strictly forward-looking, never
shuffled) and reports mean +/- std, plus bootstrap CIs and a McNemar test
against a baseline model's predictions on the same fold.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              precision_recall_curve, auc)
from statsmodels.stats.contingency_tables import mcnemar


def make_rolling_folds(all_weeks, n_folds=4, min_train_weeks=40, test_span=8):
    """
    Produces expanding-window chronological folds, e.g. for 72 weeks / 4 folds:
      fold1: train 1..40, test 41..48
      fold2: train 1..48, test 49..56
      fold3: train 1..56, test 57..64
      fold4: train 1..64, test 65..72
    """
    weeks = sorted(all_weeks)
    folds = []
    train_end = min_train_weeks
    for _ in range(n_folds):
        test_start = train_end + 1
        test_end = min(train_end + test_span, weeks[-1])
        if test_start > weeks[-1]:
            break
        folds.append((list(range(1, train_end + 1)), list(range(test_start, test_end + 1))))
        train_end = test_end
    return folds


def compute_metrics(y_true, y_pred, y_proba):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    if y_true.sum() > 0:
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_proba)
        auprc = auc(rec_curve, prec_curve)
    else:
        auprc = np.nan
    return dict(precision=p, recall=r, f1=f1, auprc=auprc)


def bootstrap_ci(y_true, y_pred, y_proba, metric="f1", n_boot=1000, seed=42, alpha=0.05):
    """Percentile bootstrap CI for a chosen metric, resampling user-weeks with replacement."""
    rng = np.random.default_rng(seed)
    y_true, y_pred, y_proba = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_proba)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
        if yt.sum() == 0:
            continue
        m = compute_metrics(yt, yp, ypr)
        stats.append(m[metric])
    stats = np.array(stats)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(stats)), float(lo), float(hi)


def mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar's test comparing two classifiers' predictions on the SAME test
    set. Table cells: both-correct, A-only-correct, B-only-correct, both-wrong.
    """
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)
    n00 = int(np.sum(~a_correct & ~b_correct))
    n01 = int(np.sum(~a_correct & b_correct))   # B right, A wrong
    n10 = int(np.sum(a_correct & ~b_correct))   # A right, B wrong
    n11 = int(np.sum(a_correct & b_correct))
    table = [[n11, n10], [n01, n00]]
    result = mcnemar(table, exact=(n01 + n10 < 25), correction=True)
    return dict(statistic=float(result.statistic), pvalue=float(result.pvalue), table=table)


def summarize_folds(fold_metric_dicts):
    df = pd.DataFrame(fold_metric_dicts)
    summary = df.agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]
    return summary
