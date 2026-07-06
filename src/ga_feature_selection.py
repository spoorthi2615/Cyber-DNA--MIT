"""
Genetic-algorithm feature selection (soft-computing component).

Why this matters for the paper: the original version picks feature BLOCKS
(USB / workstation-diversity / off-hours) by hand and adds them one at a
time (Table 3). That's a manual ablation, not feature selection. Here a GA
searches the individual-feature subset space directly, which (a) usually
finds a better-performing subset than hand-picked blocks, and (b) gives the
paper a genuine evolutionary-computation / soft-computing contribution to
point to for CISCom, instead of just asserting the XGBoost-as-soft-computing
framing.

IMPORTANT (leakage): fitness for each candidate chromosome is computed using
ONLY the training partition, via an internal chronological (not random)
K-fold split on weeks 1..train_week_cutoff. The held-out test partition is
never touched during the GA search. This mirrors the "train-only" discipline
used for normalization in the original paper.
"""
import numpy as np
import random
from deap import base, creator, tools, algorithms
import xgboost as xgb
from sklearn.metrics import f1_score

# DEAP requires module-level creator classes (only define once)
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


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


def _fitness(individual, train_df, feature_cols, label_col, n_folds=3, min_features=4):
    mask = np.array(individual, dtype=bool)
    if mask.sum() < min_features:
        return (0.0,)
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
        return (0.0,)
    # small parsimony penalty so the GA prefers compact feature sets when F1 ties
    penalty = 0.001 * mask.sum()
    return (float(np.mean(scores)) - penalty,)


def run_ga_feature_selection(train_df, feature_cols, label_col="label",
                              pop_size=30, n_gen=15, cxpb=0.6, mutpb=0.3,
                              n_folds=3, seed=42, verbose=True):
    """
    Returns (selected_feature_list, best_fitness, logbook_dataframe)
    """
    random.seed(seed)
    np.random.seed(seed)
    n_feat = len(feature_cols)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_feat)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", _fitness, train_df=train_df, feature_cols=feature_cols,
                      label_col=label_col, n_folds=n_folds)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_gen,
        stats=stats, halloffame=hof, verbose=verbose,
    )

    best = hof[0]
    selected = [f for f, m in zip(feature_cols, best) if m]
    return selected, best.fitness.values[0], logbook
