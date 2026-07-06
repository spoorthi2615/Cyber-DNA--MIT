"""
Synthetic CERT-r4.2-shaped dataset generator.

This produces WEEKLY per-user behavioral features that mirror the structure
of the real CMU CERT r4.2 dataset (logon/device/email/http derived features),
with a small injected fraction of "malicious" weeks (insider scenarios:
data exfiltration via USB, after-hours access spikes, workstation hopping).

USE: this exists so the full pipeline (features -> drift metrics -> GA
feature selection -> XGBoost + Optuna -> LSTM baseline -> fuzzy risk tiers ->
rolling-origin CV -> bootstrap CI -> SHAP) can be run and validated end-to-end
right now. When the real CERT r4.2 CSVs (logon.csv, device.csv, email.csv,
http.csv, psychometric.csv) are supplied, use data_loader.py instead, which
produces a dataframe with the IDENTICAL schema this generator produces, so
nothing downstream needs to change.
"""
import numpy as np
import pandas as pd

RNG_SEED = 42

FEATURE_COLUMNS = [
    # logon features
    "login_freq", "unique_workstations", "after_hours_logins", "failed_logins",
    "weekend_logon_ratio", "new_pc_count",
    # device features
    "usb_transfers", "usb_active_days", "usb_after_hours_count", "usb_file_count",
    # email features
    "email_sent_count", "email_recv_count", "email_ext_recipient_ratio",
    "email_attachment_count", "avg_email_size_kb", "email_after_hours_ratio",
    # http features
    "http_requests", "http_unique_domains", "http_upload_bytes", "http_job_search_hits",
    # temporal / activity
    "weekend_activity", "after_hours_ratio", "session_duration_avg", "idle_time_avg",
    # role / device diversity
    "role_change_flag", "dept_peer_similarity",
    # longitudinal (computed later, placeholders here so column count matches paper's 29)
    "BDS", "IDP", "BC", "SRC",
]
RAW_FEATURE_COUNT = len(FEATURE_COLUMNS)  # raw behavioral features + 4 longitudinal (BDS/IDP/BC/SRC)


def _base_profile(rng, n_users):
    """Each user gets a stable baseline behavior profile (role archetype)."""
    roles = rng.choice(["engineer", "sales", "finance", "admin", "exec"], size=n_users,
                        p=[0.4, 0.25, 0.15, 0.15, 0.05])
    profiles = pd.DataFrame({"user_id": [f"U{i:04d}" for i in range(n_users)], "role": roles})
    return profiles


def generate_cert_like_dataset(n_users=1000, n_weeks=72, malicious_frac=0.0048,
                                extra_source="none", seed=RNG_SEED):
    """
    Returns a long dataframe: one row per (user, week), with 29 raw behavioral
    features + user_id + week + label (1 = malicious insider activity that week).

    extra_source: "none" | "wifi_geolocation" | "psychometric"
        Optional supplementary dataset fusion (simulates adding an extra data
        modality on top of CERT r4.2, e.g. badge/geo logs or OCEAN psychometric
        scores per user) to test whether extra data sources strengthen the
        framework further, as you asked.
    """
    rng = np.random.default_rng(seed)
    profiles = _base_profile(rng, n_users)

    role_mult = {
        "engineer": dict(usb=1.3, after_hours=1.4, email=0.8, http=1.2),
        "sales": dict(usb=0.8, after_hours=0.8, email=1.4, http=1.1),
        "finance": dict(usb=0.6, after_hours=0.6, email=1.1, http=0.8),
        "admin": dict(usb=1.0, after_hours=0.7, email=1.0, http=0.9),
        "exec": dict(usb=0.5, after_hours=1.0, email=1.6, http=0.7),
    }

    rows = []
    n_malicious_target = int(round(malicious_frac * n_users * n_weeks))
    malicious_user_weeks = set()
    malicious_users = rng.choice(profiles["user_id"], size=max(1, n_malicious_target // 3), replace=False)
    for u in malicious_users:
        start_week = rng.integers(10, n_weeks - 5)
        span = rng.integers(2, 6)
        for w in range(start_week, min(start_week + span, n_weeks + 1)):
            malicious_user_weeks.add((u, w))
    # trim/pad to hit target count approximately
    malicious_user_weeks = set(list(malicious_user_weeks)[:max(1, n_malicious_target)])

    for _, prof in profiles.iterrows():
        u = prof["user_id"]
        mult = role_mult[prof["role"]]
        for w in range(1, n_weeks + 1):
            is_mal = (u, w) in malicious_user_weeks
            noise = rng.normal(1.0, 0.15, size=6).clip(0.4, 2.0)

            login_freq = rng.poisson(20 * mult["after_hours"]) * noise[0]
            unique_workstations = rng.poisson(1.5) + (2 if is_mal else 0)
            after_hours_logins = rng.poisson(2 * mult["after_hours"]) + (rng.poisson(6) if is_mal else 0)
            failed_logins = rng.poisson(0.3)
            weekend_logon_ratio = np.clip(rng.beta(1, 12) + (0.15 if is_mal else 0), 0, 1)
            new_pc_count = rng.poisson(0.2) + (1 if is_mal and rng.random() < 0.5 else 0)

            usb_transfers = rng.poisson(0.8 * mult["usb"]) + (rng.poisson(4) if is_mal else 0)
            usb_active_days = min(7, rng.poisson(0.5) + (rng.poisson(2) if is_mal else 0))
            usb_after_hours_count = rng.poisson(0.1) + (rng.poisson(3) if is_mal else 0)
            usb_file_count = rng.poisson(3 * mult["usb"]) + (rng.poisson(25) if is_mal else 0)

            email_sent_count = rng.poisson(8 * mult["email"]) * noise[1]
            email_recv_count = rng.poisson(15 * mult["email"]) * noise[2]
            email_ext_recipient_ratio = np.clip(rng.beta(2, 8) + (0.1 if is_mal else 0), 0, 1)
            email_attachment_count = rng.poisson(1.5) + (rng.poisson(3) if is_mal else 0)
            avg_email_size_kb = rng.gamma(2, 50) * noise[3]
            email_after_hours_ratio = np.clip(rng.beta(1, 10) + (0.1 if is_mal else 0), 0, 1)

            http_requests = rng.poisson(50 * mult["http"]) * noise[4]
            http_unique_domains = rng.poisson(6) + (rng.poisson(3) if is_mal else 0)
            http_upload_bytes = rng.gamma(1.5, 2000) * (5 if is_mal else 1)
            http_job_search_hits = rng.poisson(0.05) + (rng.poisson(1) if is_mal and rng.random() < 0.3 else 0)

            weekend_activity = np.clip(rng.beta(1, 15) + (0.2 if is_mal else 0), 0, 1) * noise[5]
            after_hours_ratio = np.clip(rng.beta(1, 8) + (0.15 if is_mal else 0), 0, 1)
            session_duration_avg = rng.gamma(4, 20)
            idle_time_avg = rng.gamma(3, 10)

            role_change_flag = 1 if (rng.random() < 0.002) else 0
            dept_peer_similarity = np.clip(rng.beta(6, 2) - (0.2 if is_mal else 0), 0, 1)

            row = dict(
                user_id=u, week=w, role=prof["role"], label=int(is_mal),
                login_freq=login_freq, unique_workstations=unique_workstations,
                after_hours_logins=after_hours_logins, failed_logins=failed_logins,
                weekend_logon_ratio=weekend_logon_ratio, new_pc_count=new_pc_count,
                usb_transfers=usb_transfers, usb_active_days=usb_active_days,
                usb_after_hours_count=usb_after_hours_count, usb_file_count=usb_file_count,
                email_sent_count=email_sent_count, email_recv_count=email_recv_count,
                email_ext_recipient_ratio=email_ext_recipient_ratio,
                email_attachment_count=email_attachment_count, avg_email_size_kb=avg_email_size_kb,
                email_after_hours_ratio=email_after_hours_ratio,
                http_requests=http_requests, http_unique_domains=http_unique_domains,
                http_upload_bytes=http_upload_bytes, http_job_search_hits=http_job_search_hits,
                weekend_activity=weekend_activity, after_hours_ratio=after_hours_ratio,
                session_duration_avg=session_duration_avg, idle_time_avg=idle_time_avg,
                role_change_flag=role_change_flag, dept_peer_similarity=dept_peer_similarity,
            )
            rows.append(row)

    df = pd.DataFrame(rows)

    if extra_source == "wifi_geolocation":
        # simulates fusing an extra modality: badge/wifi geo-location entropy per user-week
        df["geo_entropy"] = rng.gamma(2, 0.4, size=len(df))
        df.loc[df["label"] == 1, "geo_entropy"] += rng.gamma(2, 0.6, size=(df["label"] == 1).sum())
    elif extra_source == "psychometric":
        # simulates fusing OCEAN-like static psychometric scores per user (constant across weeks)
        psych = pd.DataFrame({
            "user_id": profiles["user_id"],
            "psych_neuroticism": rng.beta(2, 2, size=n_users),
            "psych_conscientiousness": rng.beta(2, 2, size=n_users),
        })
        df = df.merge(psych, on="user_id", how="left")

    return df.sort_values(["user_id", "week"]).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_cert_like_dataset()
    print(df.shape)
    print(df["label"].sum(), "malicious user-weeks out of", len(df))
    df.to_pickle("/home/claude/cyberdna/data/synthetic_cert_like.pkl")
