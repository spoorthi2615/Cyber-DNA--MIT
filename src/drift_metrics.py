"""
Leak-free normalization + longitudinal drift metrics (BDS, IDP, BC, SRC).

Key correctness requirement carried over from the paper (and the thing
reviewers will check first): min/max for normalization, and the "baseline
week" T_base for BDS, must be computed ONLY from weeks available up to the
current fold's training cutoff. This module is written so it can be called
fresh inside every rolling-origin fold rather than once globally, which
closes a subtle leakage gap the single-split version of the paper didn't
have to worry about.
"""
import numpy as np
import pandas as pd


def train_only_minmax(df, raw_cols, train_week_cutoff):
    """
    Fit min-max scaler using only rows with week <= train_week_cutoff,
    apply (with clipping to [0,1]) to the full dataframe.
    """
    train_mask = df["week"] <= train_week_cutoff
    mins = df.loc[train_mask, raw_cols].min()
    maxs = df.loc[train_mask, raw_cols].max()
    span = (maxs - mins).replace(0, 1.0)  # avoid divide-by-zero for constant features

    out = df.copy()
    for c in raw_cols:
        out[c] = ((df[c] - mins[c]) / span[c]).clip(0, 1)
    return out, mins, maxs


def compute_drift_metrics(df, norm_cols, id_col="user_id", week_col="week"):
    """
    df must already contain MIN-MAX NORMALIZED columns in `norm_cols`.
    Computes, per user:
      BDS(u,t)  = ||DBS_t - DBS_Tbase||_2          (deviation from earliest active week)
      IDP(u)    = 1 - mean_t ||DBS_t - DBS_{t-1}||_2   (identity persistence, per-user scalar)
      BC(u)     = 1 - Var(BDS_u,2..T)               (behavioral continuity, per-user scalar)
      SRC(u)    = 1 - mean_t ||S_t - S_{t-1}||_2     (social-role consistency; uses email-ratio
                                                        columns as the interaction-footprint vector S)
    IDP/BC/SRC are per-user scalars in the paper; we broadcast them onto every
    week-row for that user so they can be used as model features, matching
    Table/Section 5.3's "longitudinal features" treatment.
    """
    df = df.sort_values([id_col, week_col]).reset_index(drop=True)
    vecs = df[norm_cols].values

    bds = np.zeros(len(df))
    idp_map, bc_map, src_map = {}, {}, {}

    # social-role-consistency uses the email-related subset as interaction footprint S
    s_cols = [c for c in norm_cols if c.startswith("email_")]
    s_cols = s_cols if s_cols else norm_cols

    for u, g in df.groupby(id_col, sort=False):
        idx = g.index.values
        v = vecs[idx]
        t_base = 0  # earliest active week for this user (first row after sort)
        diffs = np.linalg.norm(v - v[t_base], axis=1)
        bds[idx] = diffs

        if len(v) > 1:
            step_diffs = np.linalg.norm(np.diff(v, axis=0), axis=1)
            idp = 1 - step_diffs.mean()
            bc = 1 - np.var(diffs[1:]) if len(diffs) > 1 else 1.0

            s = g[s_cols].values
            s_step_diffs = np.linalg.norm(np.diff(s, axis=0), axis=1)
            src = 1 - s_step_diffs.mean()
        else:
            idp, bc, src = 1.0, 1.0, 1.0

        idp_map[u] = idp
        bc_map[u] = bc
        src_map[u] = src

    df["BDS"] = bds
    df["IDP"] = df[id_col].map(idp_map)
    df["BC"] = df[id_col].map(bc_map)
    df["SRC"] = df[id_col].map(src_map)
    return df


def build_dbs_features(df, raw_cols, train_week_cutoff):
    """
    Full Stage 2-4 pipeline for one fold: leak-free normalization -> DBS
    construction -> drift metrics. Returns df with normalized raw_cols plus
    BDS/IDP/BC/SRC columns appended.
    """
    norm_df, mins, maxs = train_only_minmax(df, raw_cols, train_week_cutoff)
    norm_df = compute_drift_metrics(norm_df, raw_cols)
    return norm_df
