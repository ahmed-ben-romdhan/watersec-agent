"""
tools.py — All callable tools available to the WaterSec agent.
Each function returns either a dict (for text/table responses)
or a plotly Figure (for charts). The agent loop handles both.
"""

import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests as _requests
from datetime import datetime, timedelta, timezone
from scipy import stats as scipy_stats

# Will be set by app.py after data loads
DF: pd.DataFrame = None

# ISO 24512 standards cache
_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iso24512_sonede.json")
_STANDARDS = None


def _load_standards() -> dict:
    global _STANDARDS
    if _STANDARDS is None:
        with open(_JSON_PATH, "r", encoding="utf-8") as f:
            _STANDARDS = json.load(f)
    return _STANDARDS


def _get_df() -> pd.DataFrame:
    if DF is None:
        raise RuntimeError("Data not loaded yet.")
    return DF


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — query_data
# ─────────────────────────────────────────────────────────────────────────────
def query_data(
    customer: str = None,
    device_label: str = None,
    sub_category: str = None,
    start_date: str = None,
    end_date: str = None,
    group_by: str = "day",
    metric: str = "sum",
) -> dict:
    """
    Filter and aggregate water consumption data.

    Args:
        customer:     'gym' | 'customerA' | 'customerB' | 'customerC'
        device_label: human label e.g. 'Gym Cabin 1 - Hot'
        sub_category: 'Flush' | 'Sink' | 'Tap' | 'Hot' | 'Cold' etc.
        start_date:   'YYYY-MM-DD'
        end_date:     'YYYY-MM-DD'
        group_by:     'hour' | 'day' | 'week' | 'month'
        metric:       'sum' | 'mean' | 'max' | 'min' | 'count'

    Returns dict with 'table' (markdown string) and 'records' (list of dicts).
    """
    df = _get_df().copy()

    if customer:
        df = df[df["customer"].str.lower() == customer.lower()]
    if device_label:
        df = df[df["device_label"].str.contains(device_label, case=False, na=False)]
    if sub_category:
        df = df[df["sub_category_name"].str.contains(sub_category, case=False, na=False)]
    if start_date:
        df = df[df["data_time"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df = df[df["data_time"] <= pd.Timestamp(end_date + " 23:59:59", tz="UTC")]

    if df.empty:
        return {"table": "No data found for the given filters.", "records": []}

    freq_map = {"hour": "h", "day": "D", "week": "W", "month": "ME"}
    freq = freq_map.get(group_by, "D")

    grouped = (
        df.set_index("data_time")[["consumption_L", "flow_rate_Lpm"]]
        .resample(freq)
        .agg({"consumption_L": metric, "flow_rate_Lpm": "mean"})
        .reset_index()
        .rename(columns={"data_time": "period"})
    )
    grouped["period"] = grouped["period"].dt.strftime("%Y-%m-%d")
    grouped["consumption_L"] = grouped["consumption_L"].round(2)
    grouped["flow_rate_Lpm"] = grouped["flow_rate_Lpm"].round(3)
    return {
        "table": grouped.to_markdown(index=False),
        "records": grouped.to_dict(orient="records"),
        "total_L": round(df["consumption_L"].sum(), 2),
        "num_readings": len(df),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — compare_customers
# ─────────────────────────────────────────────────────────────────────────────
def compare_customers(
    customers: list = None,
    start_date: str = None,
    end_date: str = None,
    group_by: str = "month",
    metric: str = "sum",
) -> dict:
    """
    Compare total consumption across multiple customers or all customers.

    Args:
        customers:  list e.g. ['gym', 'customerA'] — omit for all
        start_date: 'YYYY-MM-DD'
        end_date:   'YYYY-MM-DD'
        group_by:   'day' | 'week' | 'month'
        metric:     'sum' | 'mean'

    Returns dict with comparison table and per-customer totals.
    """
    df = _get_df().copy()

    if customers:
        df = df[df["customer"].isin([c.lower() for c in customers])]
    if start_date:
        df = df[df["data_time"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df = df[df["data_time"] <= pd.Timestamp(end_date + " 23:59:59", tz="UTC")]

    freq_map = {"day": "D", "week": "W", "month": "ME"}
    freq = freq_map.get(group_by, "ME")

    pivot = (
        df.groupby(["customer", pd.Grouper(key="data_time", freq=freq)])["consumption_L"]
        .agg(metric)
        .reset_index()
        .rename(columns={"data_time": "period", "consumption_L": f"{metric}_L"})
    )
    pivot["period"] = pivot["period"].dt.strftime("%Y-%m")
    pivot[f"{metric}_L"] = pivot[f"{metric}_L"].round(2)

    totals = df.groupby("customer")["consumption_L"].sum().round(2).to_dict()

    return {
        "table": pivot.pivot(index="period", columns="customer", values=f"{metric}_L")
                      .fillna(0).round(2).to_markdown(),
        "totals": totals,
        "records": pivot.to_dict(orient="records"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — detect_anomalies
# ─────────────────────────────────────────────────────────────────────────────
def detect_anomalies(
    customer: str = None,
    device_label: str = None,
    window_days: int = 7,
    threshold_sigma: float = 2.5,
    start_date: str = None,
    end_date: str = None,
) -> dict:
    """
    Detect anomalous consumption spikes using rolling z-score.

    Args:
        customer:         filter to one customer
        device_label:     filter to one device
        window_days:      rolling window size in days
        threshold_sigma:  z-score threshold to flag (default 2.5)
        start_date/end_date: optional date range

    Returns dict with anomaly table and count.
    """
    df = _get_df().copy()

    if customer:
        df = df[df["customer"].str.lower() == customer.lower()]
    if device_label:
        df = df[df["device_label"].str.contains(device_label, case=False, na=False)]
    if start_date:
        df = df[df["data_time"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df = df[df["data_time"] <= pd.Timestamp(end_date + " 23:59:59", tz="UTC")]

    if df.empty:
        return {"table": "No data found.", "anomalies": [], "count": 0}

    results = []
    for dev, grp in df.groupby("device_label"):
        grp = grp.set_index("data_time").sort_index()
        roll = grp["consumption_L"].rolling(f"{window_days}D", min_periods=3)
        grp["roll_mean"] = roll.mean()
        grp["roll_std"]  = roll.std()
        grp["z_score"]   = (
            (grp["consumption_L"] - grp["roll_mean"]) / grp["roll_std"]
        ).fillna(0)
        flagged = grp[grp["z_score"].abs() > threshold_sigma].reset_index()
        flagged["device_label"] = dev
        results.append(flagged)

    if not results:
        return {"table": "No anomalies detected.", "anomalies": [], "count": 0}

    out = pd.concat(results)[
        ["data_time", "device_label", "consumption_L", "roll_mean", "z_score"]
    ].copy()
    out["data_time"]   = out["data_time"].dt.strftime("%Y-%m-%d %H:%M")
    out["roll_mean"]   = out["roll_mean"].round(2)
    out["consumption_L"] = out["consumption_L"].round(2)
    out["z_score"]     = out["z_score"].round(2)
    out = out.sort_values("z_score", ascending=False)

    return {
        "table": out.head(20).to_markdown(index=False),
        "count": len(out),
        "anomalies": out.head(20).to_dict(orient="records"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4 — detect_patterns (CustomerC sequence detection)
# ─────────────────────────────────────────────────────────────────────────────
def detect_patterns(window_minutes: int = 5) -> dict:
    """
    Detect behavioural sequences in CustomerC data.
    e.g. Flush → Sink chains within a time window.

    Uses event_start (= data_time - data_period) as the true start of each
    usage event, so sequence detection is based on when the user *opened*
    the tap, not when the sensor *sent* the data.

    Args:
        window_minutes: how long after event A ends to still count event B as a follow-up

    Returns dict with pattern counts and description.
    """
    window_minutes = int(window_minutes)
    df = _get_df().copy()
    df = df[df["customer"] == "customerC"].copy()

    if df.empty:
        return {"table": "No CustomerC data.", "patterns": {}}

    # Reconstruct real event boundaries
    # data_time  = when the sensor SENT the reading = event END
    # data_period = how many seconds the water was flowing = event DURATION
    # event_start = when the user actually opened the tap
    df["event_end"]   = df["data_time"]
    df["event_start"] = df["data_time"] - pd.to_timedelta(df["data_period"], unit="s")
    df = df.sort_values("event_start").reset_index(drop=True)

    window = pd.Timedelta(minutes=window_minutes)
    events = df[["event_start", "event_end", "sub_category_name"]].reset_index(drop=True)

    pattern_counts = {}
    # Use numpy arrays for speed instead of iterrows
    starts = events["event_start"].values.astype(np.int64)  # nanoseconds
    ends = events["event_end"].values.astype(np.int64)
    cats = events["sub_category_name"].values
    window_ns = int(window.total_seconds() * 1e9)  # convert to nanoseconds

    for i in range(len(events)):
        mask = (starts > ends[i]) & (starts <= ends[i] + window_ns)
        for j in np.where(mask)[0]:
            pair = f"{cats[i]} → {cats[j]}"
            pattern_counts[pair] = pattern_counts.get(pair, 0) + 1

    if not pattern_counts:
        return {"table": "No patterns found.", "patterns": {}}

    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])
    result_df = pd.DataFrame(sorted_patterns, columns=["sequence", "occurrences"])
    total = result_df["occurrences"].sum()
    result_df["frequency_%"] = (result_df["occurrences"] / total * 100).round(1)

    return {
        "table": result_df.head(10).to_markdown(index=False),
        "patterns": dict(sorted_patterns[:10]),
        "window_minutes": window_minutes,
        "insight": (
            f"Most common sequence: '{sorted_patterns[0][0]}' "
            f"({sorted_patterns[0][1]} times within a {window_minutes}-min window after the preceding event)"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 5 — generate_chart
# ─────────────────────────────────────────────────────────────────────────────
def generate_chart(
    customer: str = None,
    device_label: str = None,
    sub_category: str = None,
    chart_type: str = "line",
    group_by: str = "day",
    metric: str = "sum",
    start_date: str = None,
    end_date: str = None,
    title: str = "",
) -> go.Figure:
    """
    Generate a Plotly chart for consumption data.

    Args:
        chart_type: 'line' | 'bar' | 'heatmap' | 'compare'
        All other args same as query_data.

    Returns a Plotly Figure object.
    """
    df = _get_df().copy()

    if customer:
        df = df[df["customer"].str.lower() == customer.lower()]
    if device_label:
        df = df[df["device_label"].str.contains(device_label, case=False, na=False)]
    if sub_category:
        df = df[df["sub_category_name"].str.contains(sub_category, case=False, na=False)]
    if start_date:
        df = df[df["data_time"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df = df[df["data_time"] <= pd.Timestamp(end_date + " 23:59:59", tz="UTC")]

    freq_map = {"hour": "h", "day": "D", "week": "W", "month": "ME"}
    freq = freq_map.get(group_by, "D")

    if chart_type == "compare":
        # Multi-line: one line per device_label
        agg = (
            df.groupby(["device_label", pd.Grouper(key="data_time", freq=freq)])
            ["consumption_L"].agg(metric).reset_index()
        )
        fig = px.line(
            agg, x="data_time", y="consumption_L", color="device_label",
            title=title or f"Consumption comparison ({metric}, by {group_by})",
            labels={"consumption_L": "Litres", "data_time": "Date"},
        )
    elif chart_type == "heatmap":
        df["hour"] = df["data_time"].dt.hour
        df["dow"]  = df["data_time"].dt.day_name()
        pivot = df.pivot_table(
            index="dow", columns="hour", values="consumption_L", aggfunc=metric
        ).fillna(0)
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])
        fig = px.imshow(
            pivot, aspect="auto", color_continuous_scale="Blues",
            title=title or "Consumption heatmap (day of week × hour)",
            labels={"x": "Hour of day", "y": "Day", "color": "Litres"},
        )
    elif chart_type == "bar":
        agg = (
            df.set_index("data_time")["consumption_L"]
            .resample(freq).agg(metric).reset_index()
        )
        fig = px.bar(
            agg, x="data_time", y="consumption_L",
            title=title or f"Consumption ({metric}, by {group_by})",
            labels={"consumption_L": "Litres", "data_time": "Date"},
        )
    else:  # line (default)
        agg = (
            df.set_index("data_time")["consumption_L"]
            .resample(freq).agg(metric).reset_index()
        )
        fig = px.line(
            agg, x="data_time", y="consumption_L",
            title=title or f"Consumption trend ({metric}, by {group_by})",
            labels={"consumption_L": "Litres", "data_time": "Date"},
        )

    fig.update_layout(template="plotly_white", font_family="Arial")
    
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 6 — get_weather_correlation
# ─────────────────────────────────────────────────────────────────────────────
def get_weather_correlation(
    start_date: str = None,
    end_date: str = None,
    customer: str = "gym",
) -> dict:
    """
    Fetch historical daily temperature + rainfall for Tunis from Open-Meteo
    (free, no API key) and correlate with water consumption.

    Args:
        start_date: 'YYYY-MM-DD'  — defaults to 30 days ago
        end_date:   'YYYY-MM-DD'  — defaults to today
        customer:   which customer to correlate (default: gym — most weather-sensitive)

    Returns dict with correlation table, Pearson r, and insight text.
    """
    today = datetime.now(timezone.utc).date()
    if not end_date:
        end_date = str(today)
    if not start_date:
        start_date = str(today - timedelta(days=30))

    weather_df = None
    for url in [
        (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude=36.8065&longitude=10.1815"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&timezone=Africa%2FTunis"
        ),
        (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude=36.8065&longitude=10.1815"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&timezone=Africa%2FTunis&past_days=90"
        ),
    ]:
        try:
            resp = _requests.get(url, timeout=10)
            if resp.status_code == 200:
                daily = resp.json().get("daily", {})
                weather_df = pd.DataFrame({
                    "date":          pd.to_datetime(daily["time"]),
                    "temp_max":      daily["temperature_2m_max"],
                    "temp_min":      daily["temperature_2m_min"],
                    "precipitation": daily.get("precipitation_sum", [0] * len(daily["time"])),
                })
                weather_df = weather_df[
                    weather_df["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
                ]
                if not weather_df.empty:
                    break
        except Exception:
            continue

    if weather_df is None or weather_df.empty:
        return {
            "table":   "Open-Meteo API unavailable — check internet connection.",
            "records": [],
            "insight": "Weather data unavailable.",
        }

    df = _get_df().copy()
    if customer:
        df = df[df["customer"].str.lower() == customer.lower()]
    df = df[df["data_time"] >= pd.Timestamp(start_date, tz="UTC")]
    df = df[df["data_time"] <= pd.Timestamp(end_date + " 23:59:59", tz="UTC")]

    if df.empty:
        return {"table": "No consumption data for this period.", "records": [], "insight": ""}

    daily_cons = (
        df.set_index("data_time")["consumption_L"]
        .resample("D").sum()
        .reset_index()
    )
    daily_cons["date"] = daily_cons["data_time"].dt.tz_localize(None).dt.normalize()

    merged = pd.merge(daily_cons, weather_df, on="date", how="inner").dropna(
        subset=["consumption_L", "temp_max"]
    )

    if len(merged) < 5:
        return {"table": "Need ≥5 overlapping days.", "records": [], "insight": ""}

    r_temp = float(merged["consumption_L"].corr(merged["temp_max"]))
    r_rain = float(merged["consumption_L"].corr(merged["precipitation"]))

    table_df = merged.assign(
        date=merged["date"].dt.strftime("%Y-%m-%d"),
        consumption_L=merged["consumption_L"].round(1),
        temp_max_C=merged["temp_max"].round(1),
        temp_min_C=merged["temp_min"].round(1),
        rain_mm=merged["precipitation"].round(1),
    )[["date", "consumption_L", "temp_max_C", "temp_min_C", "rain_mm"]]

    if r_temp > 0.5:
        insight = (
            f"Strong positive correlation (r={r_temp:.2f}): hot days drive higher {customer} consumption. "
            f"Pre-heat water on days forecast >30°C to meet peak demand."
        )
    elif r_temp < -0.3:
        insight = f"Negative correlation (r={r_temp:.2f}): consumption drops on hot days — seasonal pattern."
    else:
        insight = f"Weak temperature correlation (r={r_temp:.2f}): {customer} demand is weather-independent."

    if r_rain < -0.3:
        insight += f" Rain reduces visits (r_rain={r_rain:.2f})."

    return {
        "table":                 table_df.to_markdown(index=False),
        "records":               table_df.to_dict(orient="records"),
        "pearson_r_temperature": round(r_temp, 3),
        "pearson_r_rainfall":    round(r_rain, 3),
        "insight":               insight,
        "data_source":           "Open-Meteo (Tunis lat=36.81, lon=10.18) — free, no API key",
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 7 — calculate_water_cost
# ─────────────────────────────────────────────────────────────────────────────
def calculate_water_cost(
    volume_L: float = None,
    customer: str = None,
    start_date: str = None,
    end_date: str = None,
    include_anomaly_cost: bool = False,
) -> dict:
    """
    Convert litres to TND using official SONEDE tariffs.

    Residential (customerC): progressive tiers 0.190 → 0.845 TND/m³ + 1.5 TND fixed fee.
    Commercial  (gym, customerA, customerB): flat 0.845 TND/m³ + 5 TND fixed fee.

    Args:
        volume_L:             direct volume in litres
        customer:             pull total from dataset if volume_L not given
        start_date/end_date:  date range for dataset query
        include_anomaly_cost: also compute TND cost of anomalous wasted events

    Returns dict with total_cost_TND, tier breakdown, and insight string.
    """
    standards = _load_standards()

    if volume_L is None and customer:
        df = _get_df().copy()
        df = df[df["customer"].str.lower() == customer.lower()]
        if start_date:
            df = df[df["data_time"] >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            df = df[df["data_time"] <= pd.Timestamp(end_date + " 23:59:59", tz="UTC")]
        volume_L = float(df["consumption_L"].sum())

    if not volume_L or volume_L <= 0:
        return {"error": "Provide volume_L or a valid customer.", "cost_TND": 0}

    volume_m3     = volume_L / 1000.0
    is_commercial = customer and customer.lower() in ["gym", "customera", "customerb"]

    if is_commercial:
        t       = standards["sonede_commercial"]
        cost    = round(volume_m3 * t["price_per_m3"] + t["fixed_fee_TND"], 3)
        details = [{
            "tier":            "Commercial flat rate",
            "volume_m3":       round(volume_m3, 3),
            "rate_TND_per_m3": t["price_per_m3"],
            "cost_TND":        round(volume_m3 * t["price_per_m3"], 3),
        }]
    else:
        tiers     = standards["sonede_tariffs"]["tiers"]
        fixed     = standards["sonede_tariffs"]["fixed_fee_TND"]
        remaining = volume_m3
        cost      = fixed
        details   = []
        for tier in tiers:
            if remaining <= 0:
                break
            in_tier   = min(remaining, tier["max_m3"] - tier["min_m3"])
            tier_cost = in_tier * tier["price_per_m3"]
            cost     += tier_cost
            details.append({
                "tier":            tier["label"],
                "volume_m3":       round(in_tier, 3),
                "rate_TND_per_m3": tier["price_per_m3"],
                "cost_TND":        round(tier_cost, 3),
            })
            remaining -= in_tier
        cost = round(cost, 3)

    anomaly_cost_TND = None
    if include_anomaly_cost and customer:
        try:
            anoms = detect_anomalies(customer=customer, start_date=start_date, end_date=end_date)
            if anoms.get("count", 0) > 0:
                waste_L  = sum(float(a["consumption_L"]) for a in anoms["anomalies"])
                rate     = (standards["sonede_commercial"]["price_per_m3"] if is_commercial
                            else standards["sonede_tariffs"]["tiers"][-1]["price_per_m3"])
                anomaly_cost_TND = round(waste_L / 1000.0 * rate, 3)
        except Exception:
            pass

    insight = (
        f"{round(volume_L):,} L = {round(volume_m3, 2)} m³ → "
        f"{round(cost, 2)} TND "
        f"({'commercial flat rate' if is_commercial else 'residential SONEDE tiered tariff'})."
    )
    if anomaly_cost_TND is not None:
        insight += f" Anomalous events waste approx. {anomaly_cost_TND} TND."

    return {
        "volume_L":         round(volume_L, 2),
        "volume_m3":        round(volume_m3, 3),
        "total_cost_TND":   round(cost, 3),
        "tariff_type":      "commercial" if is_commercial else "residential",
        "tier_breakdown":   details,
        "anomaly_cost_TND": anomaly_cost_TND,
        "insight":          insight,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 8 — check_iso24512_compliance
# ─────────────────────────────────────────────────────────────────────────────
def check_iso24512_compliance(
    customer: str = None,
    kpi_id: str = None,
) -> dict:
    """
    Evaluate ISO 24512 water service KPIs against live sensor data.

    Computable KPIs (from sensor data alone):
      W1 Non-Revenue Water (anomaly volume %)
      W4 Specific daily consumption (L/person/day)
      W5 Peak day factor (max/avg daily consumption)
      W6 Anomaly response time (WaterSec = real-time = EXCELLENT)
      W7 Metering rate (% of fixtures individually metered)
      W8 Flow rate efficiency vs fixture type benchmarks

    Not computable (reported as such):
      W2 Service continuity (needs pressure sensors)
      W3 Water quality compliance (needs quality sensors)

    Args:
        customer: 'gym' | 'customerA' | 'customerB' | 'customerC' — omit for all
        kpi_id:   'W1'–'W8' — omit for all KPIs

    Returns dict with table, results list, and summary insight.
    """
    standards = _load_standards()
    kpis      = standards["iso24512_kpis"]["kpis"]
    occupancy = standards["site_occupancy_estimates"]

    df = _get_df().copy()
    if customer:
        df = df[df["customer"].str.lower() == customer.lower()]
    if df.empty:
        return {"table": "No data.", "results": [], "insight": "No data found."}

    metering_rates = {
        "gym":       100.0,
        "customera":  50.0,
        "customerb":  11.1,
        "customerc": 100.0,
    }

    results = []

    for kpi in kpis:
        if kpi_id and kpi["id"] != kpi_id.upper():
            continue

        if not kpi.get("watersec_computable", False):
            results.append({
                "kpi_id":         kpi["id"],
                "name":           kpi["name"],
                "status":         "NOT COMPUTABLE",
                "value":          "",
                "recommendation": kpi.get("requires", "Additional sensors needed"),
            })
            continue

        status = "UNKNOWN"
        value  = None
        recommendation = ""

        try:
            if kpi["id"] == "W1":
                anoms   = detect_anomalies(customer=customer)
                total_L = float(df["consumption_L"].sum())
                if anoms["count"] > 0 and total_L > 0:
                    waste_L = sum(float(a["consumption_L"]) for a in anoms["anomalies"])
                    pct     = waste_L / total_L * 100
                    value   = round(pct, 2)
                    status  = "EXCELLENT" if pct < 10 else "ACCEPTABLE" if pct < 25 else "POOR"
                    recommendation = (
                        f"Anomalous events = {waste_L:,.1f} L ({pct:.1f}% of total). "
                        + ("Within ISO range." if pct < 25
                           else "Exceeds ISO 24512 — urgent investigation needed.")
                    )
                else:
                    value, status  = 0.0, "EXCELLENT"
                    recommendation = "No anomalies detected."

            elif kpi["id"] == "W4":
                cust_key   = (customer or "customerc").lower()
                occ        = occupancy.get(cust_key, {}).get("daily_users", 4)
                days       = max((df["data_time"].max() - df["data_time"].min()).days, 1)
                per_person = float(df["consumption_L"].sum()) / days / occ
                value      = round(per_person, 1)
                status     = "EXCELLENT" if 80 <= per_person <= 200 else "POOR"
                recommendation = (
                    f"{per_person:.1f} L/person/day (~{occ} daily users). "
                    + ("Within ISO 24512 normal range (80–200)." if status == "EXCELLENT"
                       else "Outside range — verify occupancy or check for leaks.")
                )

            elif kpi["id"] == "W5":
                daily = df.set_index("data_time")["consumption_L"].resample("D").sum()
                daily = daily[daily > 0]
                if len(daily) >= 7:
                    pdf   = float(daily.max() / daily.mean())
                    value = round(pdf, 2)
                    status = "EXCELLENT" if pdf < 1.5 else "ACCEPTABLE" if pdf < 2.5 else "POOR"
                    recommendation = (
                        f"Peak/avg = {pdf:.2f}. "
                        + ("Stable consumption." if pdf < 1.5
                           else "High variability — investigate peak events." if pdf >= 2.5
                           else "Moderate variability — normal for this site.")
                    )

            elif kpi["id"] == "W6":
                value, status  = 0.0, "EXCELLENT"
                recommendation = "WaterSec fires real-time alerts — sub-minute response latency."

            elif kpi["id"] == "W7":
                cust_key = (customer or "customerc").lower()
                rate     = metering_rates.get(cust_key, 50.0)
                value    = rate
                status   = "EXCELLENT" if rate >= 90 else "ACCEPTABLE" if rate >= 70 else "POOR"
                recommendation = (
                    f"{rate}% metering coverage. "
                    + ("Full ISO 24512 coverage." if rate >= 90
                       else "CustomerB critical gap — 1 sensor for 9+ fixtures.")
                )

            elif kpi["id"] == "W8":
                bm = kpi["benchmarks"]
                flow_results = []
                for sub in df["sub_category_name"].dropna().unique():
                    ref = next((bm[k] for k in bm if k in sub.lower()), None)
                    if not ref:
                        continue
                    mf = float(df[df["sub_category_name"] == sub]["flow_rate_Lpm"].mean())
                    rating = (
                        "efficient"     if mf <= ref["low_flow"] else
                        "normal"        if mf <= ref["target"]   else
                        "slightly high" if mf <= ref["wasteful"] else
                        "wasteful"
                    )
                    flow_results.append({"sub_category": sub, "mean_flow_Lpm": round(mf, 2), "rating": rating})
                if flow_results:
                    value  = {r["sub_category"]: r["mean_flow_Lpm"] for r in flow_results}
                    status = "POOR" if any(r["rating"] == "wasteful" for r in flow_results) else "EXCELLENT"
                    recommendation = "; ".join(
                        f"{r['sub_category']}: {r['mean_flow_Lpm']} L/min ({r['rating']})"
                        for r in flow_results
                    )

        except Exception as e:
            status, recommendation = "ERROR", str(e)

        results.append({
            "kpi_id":         kpi["id"],
            "name":           kpi["name"],
            "status":         status,
            "value":          str(value) if isinstance(value, dict) else value,
            "recommendation": recommendation,
        })

    out_df = pd.DataFrame(results)[["kpi_id", "name", "status", "value", "recommendation"]]
    n = {s: sum(1 for r in results if r["status"] == s)
         for s in ["EXCELLENT", "ACCEPTABLE", "POOR", "NOT COMPUTABLE"]}

    return {
        "table":   out_df.to_markdown(index=False),
        "results": results,
        "insight": (
            f"ISO 24512 for {customer or 'all'}: "
            f"{n['EXCELLENT']} excellent ✅  {n['ACCEPTABLE']} acceptable ⚠️  "
            f"{n['POOR']} poor ❌  {n['NOT COMPUTABLE']} need additional sensors."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 9 — session_analysis
# ─────────────────────────────────────────────────────────────────────────────
def session_analysis(customer: str = None) -> dict:
    """Per-device session statistics: avg consumption, duration, flow rate."""
    df = _get_df().copy()
    if customer:
        df = df[df["customer"].str.lower() == customer.lower()]
    if df.empty:
        return {"table": "No data.", "records": []}

    stats = (
        df.groupby(["customer", "device_label"])
        .agg(
            avg_consumption_L=("consumption_L", "mean"),
            avg_duration_s=("data_period",       "mean"),
            avg_flow_Lpm=("flow_rate_Lpm",        "mean"),
            total_events=("consumption_L",         "count"),
            total_L=("consumption_L",              "sum"),
        )
        .round(2)
        .reset_index()
    )
    return {"table": stats.to_markdown(index=False), "records": stats.to_dict(orient="records")}


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry — used by the agent loop
# ─────────────────────────────────────────────────────────────────────────────
TOOLS = {
    "query_data":                query_data,
    "compare_customers":         compare_customers,
    "detect_anomalies":          detect_anomalies,
    "detect_patterns":           detect_patterns,
    "generate_chart":            generate_chart,
    "get_weather_correlation":   get_weather_correlation,
    "calculate_water_cost":      calculate_water_cost,
    "check_iso24512_compliance": check_iso24512_compliance,
    "session_analysis":          session_analysis,
}

TOOLS_DESCRIPTION = """
Available tools (call via <tool_call> XML tag):

1. query_data — fetch and aggregate consumption for a specific customer/device/date range
   args: customer, device_label, sub_category, start_date (YYYY-MM-DD), end_date (YYYY-MM-DD),
         group_by (hour|day|week|month), metric (sum|mean|max|min|count)

2. compare_customers — compare consumption across multiple customers side by side
   args: customers (list), start_date, end_date, group_by, metric

3. detect_anomalies — find unusual consumption spikes using rolling z-score
   args: customer, device_label, window_days (int), threshold_sigma (float),
         start_date, end_date

4. detect_patterns — detect behavioural sequences in residential data (Flush→Sink chains)
   args: window_minutes (int, default 5)

5. generate_chart — produce a Plotly visual (always call for trend/comparison questions)
   args: customer, device_label, sub_category, chart_type (line|bar|heatmap|compare),
         group_by (hour|day|week|month), metric, start_date, end_date, title
   NOTE: never pass device_label="all" — omit it entirely to include all devices

6. get_weather_correlation — correlate daily consumption with Tunis temperature/rainfall
   Uses Open-Meteo free API (no key). Reveals seasonal + weather-driven patterns.
   args: start_date, end_date, customer (default: gym)
   USE WHEN: user asks about weather impact, seasonal trends, hot/cold day differences

7. calculate_water_cost — convert litres to TND using official SONEDE tariff tiers
   Residential: 4 progressive tiers 0.190–0.845 TND/m³. Commercial: flat 0.845 TND/m³.
   args: volume_L (float), customer, start_date, end_date,
         include_anomaly_cost (bool — show TND cost of wasted anomalous events)
   USE WHEN: user asks about cost, bill, TND, money, financial impact of anomalies

8. check_iso24512_compliance — evaluate site against ISO 24512 water service KPIs
   Computes W1 (leak %), W4 (L/person/day), W5 (peak day factor),
   W6 (response time), W7 (metering rate), W8 (flow rate efficiency)
   args: customer (optional), kpi_id e.g. 'W4' (optional — omit for all)
   USE WHEN: user asks about standards, compliance, KPIs, ISO benchmarks

9. session_analysis — per-device average session stats (consumption, duration, flow rate)
   args: customer (optional)
   USE WHEN: user asks about typical session, average per-device usage
"""