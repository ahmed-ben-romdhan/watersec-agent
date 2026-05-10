"""
data_loader.py — Load, clean, and merge CSV files from the data/ folder.
Handles:
  - customerA/B/C: semicolon-delimited, standard column names
  - gym: comma-delimited, dot-notation column names (data.consumption → data_consumption)
  - All files: data_consumption is in MILLILITRES → convert to LITRES
  - Filters out sensor overflow values (>10,000,000 mL = >10,000 L per event)
"""

import os
import pandas as pd
import numpy as np


def load_and_clean(data_dir: str = "data") -> pd.DataFrame:
    """
    Load all CSVs from data_dir, clean, and return a unified DataFrame.
    """
    # ── File configuration ───────────────────────────────────────────────────
    FILE_CONFIG = {
        "customerA_consumption.csv": {"customer": "customerA", "delimiter": ";"},
        "customerB_consumption.csv": {"customer": "customerB", "delimiter": ";"},
        "customerC_consumption.csv": {"customer": "customerC", "delimiter": ";"},
        "gym_consumption_data.csv":  {"customer": "gym",       "delimiter": ","},
    }

    dfs = []

    for fname, config in FILE_CONFIG.items():
        filepath = os.path.join(data_dir, fname)
        if not os.path.exists(filepath):
            print(f"[data_loader] WARNING: {fname} not found — skipping.")
            continue

        customer = config["customer"]
        delim = config["delimiter"]

        # ── Load ─────────────────────────────────────────────────────────────
        df = pd.read_csv(filepath, sep=delim, engine="python")
        print(f"[data_loader] {fname}: {len(df):,} rows")

        # ── Normalize column names (replace dots with underscores) ───────────
        df.columns = [c.strip().replace(".", "_") for c in df.columns]

        # ── Standardize column name mapping ──────────────────────────────────
        # Gym file has: data_consumption, data_time, data_period, device
        # Customer files have: device, data_consumption, data_time, data_period, tag, main_category_name, sub_category_name/type, client_id
        
        # Ensure we have the key columns
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if "consumption" in col_lower or "cons" in col_lower:
                col_map[col] = "consumption_ml"
            elif col_lower == "data_time" or col_lower == "data_time" or "time" in col_lower:
                col_map[col] = "data_time"
            elif "period" in col_lower:
                col_map[col] = "data_period"
            elif "device" in col_lower:
                col_map[col] = "device"
            elif "tag" in col_lower:
                col_map[col] = "tag"
            elif "category" in col_lower or "cat" in col_lower:
                col_map[col] = "sub_category_name"
            elif "type" in col_lower:
                col_map[col] = "type"
            elif "client" in col_lower:
                col_map[col] = "client_id"

        df.rename(columns=col_map, inplace=True)

        # ── Add customer ─────────────────────────────────────────────────────
        df["customer"] = customer

        # ── Parse timestamps ─────────────────────────────────────────────────
        if "data_time" in df.columns:
            df["data_time"] = pd.to_datetime(df["data_time"], errors="coerce")
            # Add UTC if timezone-naive
            if not df["data_time"].empty and df["data_time"].dt.tz is None:
                df["data_time"] = df["data_time"].dt.tz_localize("UTC")
        
        # Drop truly unparseable timestamps
        before = len(df)
        df = df.dropna(subset=["data_time"])
        dropped = before - len(df)
        if dropped:
            print(f"  Dropped {dropped} rows with invalid timestamps")

        # ── Convert consumption: mL → L ─────────────────────────────────────
        if "consumption_ml" in df.columns:
            df["consumption_ml"] = pd.to_numeric(df["consumption_ml"], errors="coerce")
            df["consumption_L"] = df["consumption_ml"] / 1000.0
        else:
            df["consumption_L"] = 0.0

        # ── Convert data_period to numeric ───────────────────────────────────
        if "data_period" in df.columns:
            df["data_period"] = pd.to_numeric(df["data_period"], errors="coerce").fillna(0)

        # ── Filter corrupted/sensor-overflow values (>10,000 L per event) ────
        before = len(df)
        df = df[df["consumption_L"] <= 10000]
        dropped = before - len(df)
        if dropped:
            print(f"  Dropped {dropped} rows with consumption > 10,000 L (sensor overflow)")

        # ── Calculate flow rate (L/min) ──────────────────────────────────────
        df["flow_rate_Lpm"] = np.where(
            df["data_period"] > 0,
            df["consumption_L"] / (df["data_period"] / 60.0),
            0.0,
        )

        # ── Generate device labels ───────────────────────────────────────────
        device_labels = []
        for _, row in df.iterrows():
            cust = row.get("customer", "?")
            dev_id = str(row.get("device", ""))[:8]
            sub = str(row.get("sub_category_name", ""))
            tag = str(row.get("tag", ""))
            typ = str(row.get("type", ""))

            if cust == "gym":
                # Use tag (Hot/Cold) + device ID for gym
                label = f"Gym {tag} {dev_id}" if tag else f"Gym {dev_id}"
            elif cust in ("customerA",):
                label = f"Office Bloc {dev_id}"
            elif cust == "customerB":
                label = f"Sanitary Bloc {dev_id}"
            elif cust == "customerC":
                label = f"Residential {sub} {tag}".strip()
            else:
                label = str(row.get("device", ""))[:16]

            device_labels.append(label)

        df["device_label"] = device_labels

        dfs.append(df)

    # ── Merge all DataFrames ─────────────────────────────────────────────────
    if not dfs:
        raise FileNotFoundError(f"No CSV files loaded from '{data_dir}/'")

    # Deduplicate columns and reset indexes before concat
    dfs = [d.loc[:, ~d.columns.duplicated()].reset_index(drop=True) for d in dfs]
    df = pd.concat(dfs, ignore_index=True)

    # ── Final summary ────────────────────────────────────────────────────────
    total_rows = len(df)
    n_devices = df["device_label"].nunique()
    n_customers = df["customer"].nunique()
    print(f"[data_loader] Loaded {total_rows:,} rows across {n_customers} customers, {n_devices} devices.")

    return df


def get_summary(df: pd.DataFrame) -> str:
    """Generate a live dataset summary for the system prompt."""
    lines = []
    profiles = {
        "gym":       ("Shower block",       "8 sensors (4 cabins × Hot/Cold)"),
        "customerA": ("Office toilet bloc", "4 aggregated bloc sensors"),
        "customerB": ("Sanitary bloc",      "1 aggregated sensor"),
        "customerC": ("Residential home",   "7 individual sensors (Flush/Sink/Tap)"),
    }

    for customer, grp in df.groupby("customer"):
        profile, desc = profiles.get(customer.lower(), ("Unknown", "?"))
        dmin = grp["data_time"].min().strftime("%b %d %Y")
        dmax = grp["data_time"].max().strftime("%b %d %Y")
        total_L = grp["consumption_L"].sum()
        days = max((grp["data_time"].max() - grp["data_time"].min()).days, 1)
        avg_day = total_L / days
        devices = grp["device_label"].nunique()
        lines.append(
            f"- {customer} ({profile}): {desc}. "
            f"Data: {dmin} → {dmax} ({days} days). "
            f"Total: {total_L:,.0f} L. "
            f"Avg: {avg_day:,.0f} L/day. "
            f"Devices: {devices}."
        )

    return "\n".join(lines)