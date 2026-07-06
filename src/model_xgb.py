"""
XGBoost classifier tuned with Optuna (Bayesian/TPE hyperparameter search),
mirroring what reference [21] does (Bayesian-optimized XGBoost) but wired
into our strict chronological protocol: the inner tuning loop never sees
val weeks by touching future weeks during fit.
"""
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score, precision_recall_curve, auc

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _chronological_split(train_df, week_col="week", val_frac=0.2):
    weeks = sorted(train_df[week_col].unique())
    cut = int(len(weeks) * (1 - val_frac))
    fit_weeks, val_weeks = weeks[:cut], weeks[cut:]
    return (train_df[train_df[week_col].isin(fit_weeks)],
            train_df[train_df[week_col].isin(val_weeks)])


def tune_xgb(train_df, feature_cols, label_col="label", n_trials=40, seed=42):
    fit_df, val_df = _chronological_split(train_df)
    spw = (fit_df[label_col] == 0).sum() / max(1, (fit_df[label_col] == 1).sum())

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", spw * 0.5, spw * 2.0),
        )
        clf = xgb.XGBClassifier(**params, eval_metric="aucpr", n_jobs=2, verbosity=0, random_state=seed)
        clf.fit(fit_df[feature_cols], fit_df[label_col])
        proba = clf.predict_proba(val_df[feature_cols])[:, 1]
        p, r, _ = precision_recall_curve(val_df[label_col], proba)
        return auc(r, p)  # AUPRC as tuning objective (matches paper's emphasis under imbalance)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def fit_final_model(train_df, feature_cols, best_params, label_col="label", seed=42):
    clf = xgb.XGBClassifier(**best_params, eval_metric="aucpr", n_jobs=2, verbosity=0, random_state=seed)
    clf.fit(train_df[feature_cols], train_df[label_col])
    return clf


def optimize_threshold(clf, val_df, feature_cols, label_col="label", metric="f1"):
    proba = clf.predict_proba(val_df[feature_cols])[:, 1]
    best_t, best_score = 0.5, -1
    for t in np.linspace(0.05, 0.95, 37):
        preds = (proba >= t).astype(int)
        score = f1_score(val_df[label_col], preds, zero_division=0)
        if score > best_score:
            best_t, best_score = t, score
    return best_t, best_score
