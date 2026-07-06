import numpy as np
import random
import xgboost as xgb
from sklearn.metrics import f1_score
import pandas as pd

def _chronological_inner_folds(train_df, n_folds, week_col="week"):
    """Yield (fit_weeks, val_weeks) splits strictly within the training partition."""
    weeks = sorted(train_df[week_col].unique())
    n = len(weeks)
    fold_edges = np.linspace(int(n * 0.5), n, n_folds + 1).astype(int)
    for i in range(n_folds):
        fit_weeks = weeks[: fold_edges[i]]
        val_weeks = weeks[fold_edges[i]: fold_edges[i + 1]]
        if len(val_weeks) == 0:
            continue
        yield fit_weeks, val_weeks

def _fitness(mask, train_df, feature_cols, label_col, n_folds=3, min_features=4):
    mask = np.array(mask, dtype=bool)
    if mask.sum() < min_features:
        return 0.0
    chosen = [f for f, m in zip(feature_cols, mask) if m]

    scores = []
    for fit_weeks, val_weeks in _chronological_inner_folds(train_df, n_folds):
        fit_df = train_df[train_df["week"].isin(fit_weeks)]
        val_df = train_df[train_df["week"].isin(val_weeks)]
        if fit_df[label_col].sum() == 0 or val_df[label_col].sum() == 0:
            continue
        clf = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            scale_pos_weight=(fit_df[label_col] == 0).sum() / max(1, (fit_df[label_col] == 1).sum()),
            eval_metric="aucpr", n_jobs=2, verbosity=0,
        )
        clf.fit(fit_df[chosen], fit_df[label_col])
        preds = clf.predict(val_df[chosen])
        scores.append(f1_score(val_df[label_col], preds, zero_division=0))

    if not scores:
        return 0.0
    # small parsimony penalty so the algorithm prefers compact feature sets when F1 ties
    penalty = 0.001 * mask.sum()
    return float(np.mean(scores)) - penalty

def run_quantum_feature_selection(train_df, feature_cols, label_col="label",
                                  pop_size=30, n_gen=15, mutpb=0.01,
                                  n_folds=3, seed=42, verbose=True):
    """
    Quantum-Inspired Evolutionary Algorithm (QEA) based on Han & Kim (2002).
    Returns (selected_feature_list, best_fitness, logbook_dataframe)
    where logbook_dataframe is a list of dicts with keys: ['gen', 'avg', 'max']
    """
    random.seed(seed)
    np.random.seed(seed)
    n_feat = len(feature_cols)

    # Initialize Q-population: N individuals, each is an array of M qubits (alpha, beta)
    # alpha = beta = 1/sqrt(2)
    q_pop = np.full((pop_size, n_feat, 2), 1.0 / np.sqrt(2.0))

    global_best_mask = None
    global_best_fit = -1.0
    
    logbook = []
    
    theta_step = 0.05 * np.pi

    for gen in range(1, n_gen + 1):
        # 1. Observation (collapse qubits to binary masks)
        # Probability of 1 is beta^2
        probs_1 = q_pop[:, :, 1] ** 2
        rand_vals = np.random.rand(pop_size, n_feat)
        binary_pop = (rand_vals < probs_1).astype(int)

        # 2. Evaluation
        fitnesses = np.zeros(pop_size)
        for i in range(pop_size):
            fit = _fitness(binary_pop[i], train_df, feature_cols, label_col, n_folds)
            fitnesses[i] = fit
            
            # Update global best
            if fit > global_best_fit:
                global_best_fit = fit
                global_best_mask = binary_pop[i].copy()
                
        # Logging
        avg_fit = np.mean(fitnesses)
        max_fit = np.max(fitnesses)
        logbook.append({'gen': gen, 'avg': avg_fit, 'max': max_fit})
        
        if verbose:
            print(f"QEA Gen {gen}: Max Fit {max_fit:.4f}, Avg Fit {avg_fit:.4f}")

        # 3. Update Qubits (Rotation Gate)
        for i in range(pop_size):
            for j in range(n_feat):
                alpha, beta = q_pop[i, j]
                x = binary_pop[i, j]
                b = global_best_mask[j]
                
                # QEA Lookup Table logic for rotation angle theta
                theta = 0.0
                if x == 0 and b == 1:
                    # we want more 1s -> increase beta^2
                    sign = +1 if alpha * beta > 0 else -1
                    theta = sign * theta_step
                elif x == 1 and b == 0:
                    # we want more 0s -> decrease beta^2
                    sign = -1 if alpha * beta > 0 else +1
                    theta = sign * theta_step
                    
                if theta != 0.0:
                    new_alpha = np.cos(theta) * alpha - np.sin(theta) * beta
                    new_beta = np.sin(theta) * alpha + np.cos(theta) * beta
                    q_pop[i, j, 0] = new_alpha
                    q_pop[i, j, 1] = new_beta
                    
                # 4. Mutation (Pauli-X like flip)
                if random.random() < mutpb:
                    # Swap alpha and beta to flip probability amplitude
                    temp = q_pop[i, j, 0]
                    q_pop[i, j, 0] = q_pop[i, j, 1]
                    q_pop[i, j, 1] = temp

    selected = [f for f, m in zip(feature_cols, global_best_mask) if m]
    return selected, global_best_fit, logbook
