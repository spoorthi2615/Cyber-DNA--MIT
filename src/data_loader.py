"""
Real CERT r4.2 loader.

Point this at the actual r4.2 release folder (the standard CMU CERT release
contains logon.csv, device.csv, email.csv, http.csv, psychometric.csv, and an
answers/ or ground-truth insiders.csv listing which users/date-ranges are
malicious scenarios). This module aggregates raw events into the SAME
weekly per-user schema data_gen.py produces synthetically, so main.py does
not need to change -- just swap the import:

    # from data_gen import generate_cert_like_dataset
    from data_loader import load_real_cert_r42 as generate_cert_like_dataset

USAGE:
    df = load_real_cert_r42(cert_dir="/mnt/user-data/uploads/cert_r42",
                             ground_truth_path="/mnt/user-data/uploads/cert_r42/answers/insiders.csv")

NOTE ON GROUND TRUTH: CERT r4.2 ships a separate answer key identifying which
(user, date-range) combinations correspond to the synthetic insider
scenarios. You MUST supply this file's path -- without it there is no label
column and the whole supervised pipeline (GA fitness, XGBoost, thresholds,
bootstrap CI, McNemar) has nothing to train/score against. Column names
below match the standard r4.2 release; adjust the *_COLS dicts if your
extract uses different headers (some redistributions rename columns).
"""
import pandas as pd
import numpy as np

LOGON_COLS = dict(user="user", date="date", pc="pc", activity="activity")  # activity: Logon/Logoff
DEVICE_COLS = dict(user="user", date="date", pc="pc", activity="activity")  # activity: Connect/Disconnect
EMAIL_COLS = dict(user="user", date="date", to="to", cc="cc", bcc="bcc",
                   from_="from", size="size", attachments="attachments")
HTTP_COLS = dict(user="user", date="date", url="url")


def _week_of(dates):
    d = pd.to_datetime(dates)
    origin = d.min().normalize()
    return ((d - origin).dt.days // 7) + 1


def _agg_logon(logon_df):
    logon_df["week"] = _week_of(logon_df[LOGON_COLS["date"]])
    logon_df["dt"] = pd.to_datetime(logon_df[LOGON_COLS["date"]])
    logon_df["hour"] = logon_df["dt"].dt.hour
    logon_df["dow"] = logon_df["dt"].dt.dayofweek  # 5,6 = weekend

    logons = logon_df[logon_df[LOGON_COLS["activity"]].str.lower() == "logon"]
    g = logons.groupby([LOGON_COLS["user"], "week"])
    out = pd.DataFrame({
        "login_freq": g.size(),
        "unique_workstations": g[LOGON_COLS["pc"]].nunique(),
        "after_hours_logins": logons[(logons["hour"] < 7) | (logons["hour"] > 19)]
            .groupby([LOGON_COLS["user"], "week"]).size(),
        "weekend_logon_ratio": g.apply(lambda x: (x["dow"] >= 5).mean()),
    }).reset_index().rename(columns={LOGON_COLS["user"]: "user_id"})
    out["failed_logins"] = 0  # r4.2 logon.csv has no failure flag; set from auth logs if available
    out["new_pc_count"] = 0   # requires per-user PC history; compute via rolling set-diff if needed
    return out.fillna(0)


def _agg_device(device_df):
    device_df["week"] = _week_of(device_df[DEVICE_COLS["date"]])
    device_df["dt"] = pd.to_datetime(device_df[DEVICE_COLS["date"]])
    device_df["hour"] = device_df["dt"].dt.hour

    connects = device_df[device_df[DEVICE_COLS["activity"]].str.lower() == "connect"]
    g = connects.groupby([DEVICE_COLS["user"], "week"])
    out = pd.DataFrame({
        "usb_transfers": g.size(),
        "usb_active_days": g["dt"].apply(lambda x: x.dt.date.nunique()),
        "usb_after_hours_count": connects[(connects["hour"] < 7) | (connects["hour"] > 19)]
            .groupby([DEVICE_COLS["user"], "week"]).size(),
    }).reset_index().rename(columns={DEVICE_COLS["user"]: "user_id"})
    out["usb_file_count"] = 0  # requires file.csv join if available in your extract
    return out.fillna(0)


def _agg_email(email_df):
    email_df["week"] = _week_of(email_df[EMAIL_COLS["date"]])
    g = email_df.groupby([EMAIL_COLS["user"], "week"])
    out = pd.DataFrame({
        "email_sent_count": g.size(),
        "email_attachment_count": g[EMAIL_COLS["attachments"]].sum() if EMAIL_COLS["attachments"] in email_df else 0,
        "avg_email_size_kb": g[EMAIL_COLS["size"]].mean() / 1024.0 if EMAIL_COLS["size"] in email_df else 0,
    }).reset_index().rename(columns={EMAIL_COLS["user"]: "user_id"})
    out["email_recv_count"] = 0        # requires reverse join on `to` field
    out["email_ext_recipient_ratio"] = 0  # requires domain-matching on `to`/`from` vs internal domain
    out["email_after_hours_ratio"] = 0
    return out.fillna(0)


def _agg_http(http_df):
    if http_df.empty:
        return pd.DataFrame(columns=["user_id", "week", "http_requests", "http_unique_domains", "http_upload_bytes", "http_job_search_hits"])
    http_df["week"] = _week_of(http_df[HTTP_COLS["date"]])
    g = http_df.groupby([HTTP_COLS["user"], "week"])
    out = pd.DataFrame({
        "http_requests": g.size(),
        "http_unique_domains": g[HTTP_COLS["url"]].apply(lambda x: x.str.extract(r"://([^/]+)")[0].nunique()),
    }).reset_index().rename(columns={HTTP_COLS["user"]: "user_id"})
    out["http_upload_bytes"] = 0
    out["http_job_search_hits"] = g[HTTP_COLS["url"]].apply(
        lambda x: x.str.contains("job|career|indeed|linkedin", case=False, regex=True).sum()
    ).values
    return out.fillna(0)


def _attach_ground_truth(user_week_df, ground_truth_path):
    """
    ground_truth_path: CSV with at least [user, start_date, end_date] (r4.2
    answer-key format varies by scenario release -- adjust column names below
    to match your extract).
    """
    gt = pd.read_csv(ground_truth_path)
    gt_cols = {c.lower(): c for c in gt.columns}
    user_c = gt_cols.get("user", list(gt.columns)[0])
    start_c = gt_cols.get("start", gt_cols.get("start_date"))
    end_c = gt_cols.get("end", gt_cols.get("end_date"))

    user_week_df["label"] = 0
    user_week_df["_week_actual_start"] = pd.to_datetime('2010-01-02') + pd.to_timedelta((user_week_df["week"] - 1) * 7, unit="d")
    user_week_df["_week_actual_end"] = user_week_df["_week_actual_start"] + pd.to_timedelta(6, unit="d")

    # Mark malicious weeks: any week overlapping [start_c, end_c] for that user
    for _, row in gt.iterrows():
        u = row[user_c]
        s, e = pd.to_datetime(row[start_c]), pd.to_datetime(row[end_c])
        mask = (user_week_df["user_id"] == u) & \
               (user_week_df["_week_actual_start"] <= e) & (user_week_df["_week_actual_end"] >= s)
        user_week_df.loc[mask, "label"] = 1

    return user_week_df.drop(columns=["_week_actual_start", "_week_actual_end"], errors="ignore")


def load_real_cert_r42(cert_dir, ground_truth_path=None, extra_files=None):
    """
    cert_dir: folder containing logon.csv, device.csv, email.csv, http.csv
    ground_truth_path: path to the insider-scenario answer key (see above)
    extra_files: optional dict, e.g. {"psychometric": "psychometric.csv"} for
                 additional per-user static features to fuse in, or
                 {"badge": "badge.csv"} for physical access-log geo features.

    Returns a dataframe with schema: user_id, week, label, <feature columns...>
    matching data_gen.FEATURE_COLUMNS (minus BDS/IDP/BC/SRC, which are
    computed downstream by drift_metrics.build_dbs_features per fold).
    """
    import os
    logon = pd.read_csv(f"{cert_dir}/logon.csv")
    device = pd.read_csv(f"{cert_dir}/device.csv")
    email = pd.read_csv(f"{cert_dir}/email.csv")
    if os.path.exists(f"{cert_dir}/http.csv"):
        http = pd.read_csv(f"{cert_dir}/http.csv")
    else:
        http = pd.DataFrame(columns=["id", "date", "user", "pc", "url"])

    logon_feats = _agg_logon(logon)
    device_feats = _agg_device(device)
    email_feats = _agg_email(email)
    http_feats = _agg_http(http)

    merged = logon_feats
    for other in (device_feats, email_feats, http_feats):
        merged = merged.merge(other, on=["user_id", "week"], how="outer")
    merged = merged.fillna(0)

    # placeholders for features that need extra joins not shown above
    for col in ["role_change_flag", "dept_peer_similarity", "weekend_activity",
                "after_hours_ratio", "session_duration_avg", "idle_time_avg"]:
        if col not in merged.columns:
            merged[col] = 0.0

    if extra_files:
        for name, path in extra_files.items():
            extra = pd.read_csv(f"{cert_dir}/{path}")
            merged = merged.merge(extra, on="user_id", how="left")

    if ground_truth_path:
        merged = _attach_ground_truth(merged, ground_truth_path)
    else:
        merged["label"] = 0  # WARNING: no labels attached; supervised pipeline will not work until you provide one

    return merged.sort_values(["user_id", "week"]).reset_index(drop=True)
