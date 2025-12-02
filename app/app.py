# app/app.py

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NOAA NYC Climate Explorer",
    layout="wide",
)

st.title("NOAA NYC Climate Explorer (2020–2025)")

st.markdown(
    """
This dashboard summarizes **annual climate conditions and short-term trends**
for weather stations in the **New York City metropolitan area** using a subset
of the NOAA GSOM annual summaries (2020–2025).

Use the controls in the sidebar to:

- Pick the year range used for summaries and trend calculations.  
- Select the primary climate metric for comparisons.  
- Choose specific stations to examine in more detail.
"""
)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the cleaned annual dataset produced by the notebooks.

    Expected columns (at minimum):
    - STATION, NAME, DATE (year)
    - LATITUDE, LONGITUDE, ELEVATION
    - TAVG, PRCP, SNOW
    - heat_extreme_days, cold_extreme_days (if available)
    """
    df = pd.read_csv("data/processed/noaa_nyc_annual_clean.csv")
    df["DATE"] = df["DATE"].astype(int)
    return df


df = load_data()

year_min, year_max = int(df["DATE"].min()), int(df["DATE"].max())
stations_all = df["NAME"].sort_values().unique()

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("Controls")

year_range = st.sidebar.slider(
    "Select year range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
)

metric_label = st.sidebar.selectbox(
    "Primary climate metric",
    [
        "Average temperature (TAVG, °F)",
        "Total precipitation (PRCP, inches)",
        "Total snowfall (SNOW, inches)",
        "Heat extreme days (heat_extreme_days)",
        "Cold extreme days (cold_extreme_days)",
    ],
)

# Map friendly labels to column names and axis labels
METRIC_CONFIG = {
    "Average temperature (TAVG, °F)": ("TAVG", "Average temperature (°F)"),
    "Total precipitation (PRCP, inches)": ("PRCP", "Total precipitation (inches)"),
    "Total snowfall (SNOW, inches)": ("SNOW", "Total snowfall (inches)"),
    "Heat extreme days (heat_extreme_days)": ("heat_extreme_days", "Heat extreme days (≥ threshold)"),
    "Cold extreme days (cold_extreme_days)": ("cold_extreme_days", "Cold extreme days (≤ threshold)"),
}
metric_col, metric_axis_label = METRIC_CONFIG[metric_label]

# Filter years
mask_years = (df["DATE"] >= year_range[0]) & (df["DATE"] <= year_range[1])
df_filt = df.loc[mask_years].copy()

if df_filt.empty:
    st.error("No data in the selected year range.")
    st.stop()

# Guard: metric column might not exist (e.g. extremes if you didn’t compute them)
if metric_col not in df_filt.columns:
    st.error(
        f"Column **`{metric_col}`** is not present in `noaa_nyc_annual_clean.csv`.\n\n"
        "Switch to **TAVG**, **PRCP**, or **SNOW**, or add this metric in your preprocessing step."
    )
    st.stop()

stations_default = [s for s in stations_all if "CENTRAL PARK" in s] or list(stations_all[:3])

selected_stations = st.sidebar.multiselect(
    "Stations for detailed exploration",
    options=list(stations_all),
    default=stations_default,
)

top_n = st.sidebar.slider(
    "Number of stations in rankings",
    min_value=5,
    max_value=30,
    value=15,
)

st.sidebar.markdown(
    """
Trends are computed for each station using a simple least-squares line fit over
the selected year range. With only 6 years of data (2020–2025), treat these as
**short-term signals**, not long-term climate normals.
"""
)

# -----------------------------------------------------------------------------
# Helper: compute station-level trends
# -----------------------------------------------------------------------------
def compute_station_trends(df_years: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Fit a linear trend for each station for the chosen metric over the given years.
    Returns one row per station with a `slope` column (units of metric per year).
    """

    def slope_for_group(g: pd.DataFrame) -> float:
        g = g.dropna(subset=[col])
        if g.shape[0] < 3:
            return np.nan
        x = g["DATE"].values.astype(float)
        y = g[col].values.astype(float)
        m, _ = np.polyfit(x, y, 1)
        return m

    # groupby + apply returns a Series; convert to a clean DataFrame
    trend_series = df_years.groupby(["STATION", "NAME"]).apply(slope_for_group)
    trend_df = trend_series.reset_index(name="slope")
    return trend_df


# Precompute metro time series for the selected metric
metro_series = df_filt.groupby("DATE", as_index=False)[metric_col].mean()
metro_mean = metro_series[metric_col].mean()
metro_first = metro_series.loc[metro_series["DATE"].idxmin(), metric_col]
metro_last = metro_series.loc[metro_series["DATE"].idxmax(), metric_col]
delta_val = metro_last - metro_first

# -----------------------------------------------------------------------------
# 1. Overview KPIs
# -----------------------------------------------------------------------------
st.subheader("Overview for selected period")

col1, col2, col3 = st.columns(3)
col1.metric(
    f"Metro-wide mean {metric_axis_label.lower()}",
    f"{metro_mean:.2f}",
)
col2.metric(
    f"Change in metro mean ({year_range[0]} → {year_range[1]})",
    f"{delta_val:+.2f} per year-equivalent",
)
col3.metric(
    "Number of stations with data",
    f"{df_filt['STATION'].nunique()}",
)

st.markdown("---")

# -----------------------------------------------------------------------------
# 2. Spatial patterns: map of stations and anomalies
# -----------------------------------------------------------------------------
st.subheader("Spatial patterns in the selected metric")

has_geo = all(c in df_filt.columns for c in ["LATITUDE", "LONGITUDE"])

# Aggregate per station (mean over selected years, plus location info)
agg_dict = {metric_col: "mean"}
if has_geo:
    agg_dict.update({"LATITUDE": "first", "LONGITUDE": "first"})
if "ELEVATION" in df_filt.columns:
    agg_dict["ELEVATION"] = "first"

station_summary = (
    df_filt.groupby(["STATION", "NAME"], as_index=False)
    .agg(agg_dict)
)

station_summary["delta_vs_metro"] = station_summary[metric_col] - metro_mean

if has_geo:
    center_lat = station_summary["LATITUDE"].mean()
    center_lon = station_summary["LONGITUDE"].mean()

    fig_map = px.scatter_mapbox(
        station_summary,
        lat="LATITUDE",
        lon="LONGITUDE",
        size=np.maximum(station_summary[metric_col] - station_summary[metric_col].min(), 0.1),
        size_max=18,
        color="delta_vs_metro",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        hover_name="NAME",
        hover_data={
            metric_col: ":.2f",
            "delta_vs_metro": ":+.2f",
            "ELEVATION": True if "ELEVATION" in station_summary.columns else False,
        },
        title=(
            f"Station locations coloured by anomaly vs metro mean "
            f"({metric_axis_label}, {year_range[0]}–{year_range[1]})"
        ),
        zoom=7,
        center={"lat": center_lat, "lon": center_lon},
    )
    fig_map.update_layout(
        mapbox_style="carto-positron",
        height=550,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("Latitude/longitude columns not found, so the map is skipped.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 3. Ranking: Top N stations by metric (advanced bar with anomalies)
# -----------------------------------------------------------------------------
st.subheader(f"Station rankings by {metric_axis_label.lower()}")

top_ranked = (
    station_summary.nlargest(top_n, metric_col).sort_values(metric_col)
)

rank_col1, rank_col2 = st.columns([3, 2])

with rank_col1:
    fig_rank = px.bar(
        top_ranked,
        x=metric_col,
        y="NAME",
        orientation="h",
        color="delta_vs_metro",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        labels={
            "NAME": "Station",
            metric_col: metric_axis_label,
            "delta_vs_metro": f"Δ vs metro mean ({metric_axis_label})",
        },
        title=(
            f"Top {top_n} stations by {metric_axis_label.lower()} "
            f"({year_range[0]}–{year_range[1]})"
        ),
        hover_data={"delta_vs_metro": ":+.2f"},
    )
    fig_rank.add_vline(
        x=metro_mean,
        line_dash="dash",
        line_color="white",
        annotation_text="Metro mean",
        annotation_position="top right",
    )
    fig_rank.update_layout(height=550, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_rank, use_container_width=True)

with rank_col2:
    st.markdown(
        """
**How to read this chart**

- Bars show the **mean value** for each station over the selected years.
- Colour encodes how far each station is from the **metro-wide mean**:
  - values above the mean are warmer/wetter/snowier (red shades),
  - values below the mean are cooler/drier (blue shades).
- The dashed vertical line marks the metro average.

This gives a quick sense of which locations are systematic **hot spots**
or **cold spots** for the chosen metric.
"""
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. Time series for selected stations + anomaly view
# -----------------------------------------------------------------------------
st.subheader("Year-to-year evolution")

if not selected_stations:
    st.info("Select at least one station in the sidebar to see time series.")
else:
    # Data for selected stations
    df_ts = df_filt[df_filt["NAME"].isin(selected_stations)].copy()

    # Merge metro mean to compute anomalies
    metro_for_merge = metro_series.rename(columns={metric_col: "metro_mean"})
    df_ts = df_ts.merge(metro_for_merge, on="DATE", how="left")
    df_ts["anomaly"] = df_ts[metric_col] - df_ts["metro_mean"]

    # Absolute values with metro mean as an extra "station"
    metro_series_long = metro_series.copy()
    metro_series_long["NAME"] = "Metro mean (all stations)"
    df_ts_abs = pd.concat(
        [
            df_ts[["DATE", "NAME", metric_col]],
            metro_series_long[["DATE", "NAME", metric_col]],
        ],
        ignore_index=True,
    )

    abs_col, anom_col = st.columns(2)

    with abs_col:
        fig_ts_abs = px.line(
            df_ts_abs,
            x="DATE",
            y=metric_col,
            color="NAME",
            markers=True,
            labels={
                "DATE": "Year",
                "NAME": "Station",
                metric_col: metric_axis_label,
            },
            title=f"{metric_axis_label} by year for selected stations ({year_range[0]}–{year_range[1]})",
        )
        fig_ts_abs.update_layout(height=500, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_ts_abs, use_container_width=True)

    with anom_col:
        fig_ts_anom = px.line(
            df_ts,
            x="DATE",
            y="anomaly",
            color="NAME",
            markers=True,
            labels={
                "DATE": "Year",
                "NAME": "Station",
                "anomaly": f"Anomaly vs metro mean ({metric_axis_label})",
            },
            title=(
                f"Anomaly vs metro mean for selected stations "
                f"({metric_axis_label}, {year_range[0]}–{year_range[1]})"
            ),
        )
        fig_ts_anom.add_hline(
            y=0,
            line_dash="dash",
            line_color="white",
            annotation_text="Metro mean",
            annotation_position="top left",
        )
        fig_ts_anom.update_layout(height=500, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_ts_anom, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. Station-level trends (slopes) + distribution
# -----------------------------------------------------------------------------
st.subheader(f"Short-term trends in {metric_axis_label.lower()}")

trend_df = compute_station_trends(df_filt, metric_col)
trend_valid = trend_df.dropna(subset=["slope"])

if trend_valid.empty:
    st.info(
        "Not enough non-missing data to compute station-level trends for this "
        "metric and year range."
    )
else:
    warmest_trend = (
        trend_valid.nlargest(top_n, "slope").sort_values("slope")
    )

    trend_col1, trend_col2 = st.columns([3, 2])

    with trend_col1:
        fig_trend_rank = px.bar(
            warmest_trend,
            x="slope",
            y="NAME",
            orientation="h",
            color="slope",
            color_continuous_scale="Reds",
            labels={
                "NAME": "Station",
                "slope": f"Trend in {metric_axis_label.lower()} per year",
            },
            title=(
                f"Stations with strongest upward trend in "
                f"{metric_axis_label.lower()} ({year_range[0]}–{year_range[1]})"
            ),
        )
        fig_trend_rank.add_vline(
            x=0,
            line_dash="dash",
            line_color="white",
            annotation_text="No trend",
            annotation_position="top left",
        )
        fig_trend_rank.update_layout(height=550, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_trend_rank, use_container_width=True)

    with trend_col2:
        st.markdown(
            """
**What this shows**

- Each bar is the **slope of a linear regression** of the chosen metric vs. year
  for one station.
- Positive slopes indicate **increasing** values over the period
  (warming, wetter conditions, more hot days, etc.).
- The dashed vertical line at 0 marks "no trend".

Because we only have **six years of data**, these slopes should be interpreted
as **short-term signals** rather than robust long-term climate trends.
"""
        )

    # Distribution of slopes
    fig_hist = px.histogram(
        trend_valid,
        x="slope",
        nbins=20,
        labels={"slope": f"Trend in {metric_axis_label.lower()} per year"},
        title=f"Distribution of station-level trends in {metric_axis_label.lower()}",
    )
    fig_hist.add_vline(
        x=0,
        line_dash="dash",
        line_color="white",
        annotation_text="No trend",
        annotation_position="top left",
    )
    fig_hist.update_layout(height=400, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)
