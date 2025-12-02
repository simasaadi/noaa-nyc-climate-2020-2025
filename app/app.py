import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="NOAA NYC Climate Explorer (2020–2025)",
    layout="wide",
)

# ---------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load cleaned NOAA annual summary data for the NYC metro area."""
    data_path = (
        Path(__file__)
        .resolve()
        .parents[1]
        / "data"
        / "processed"
        / "noaa_nyc_annual_clean.csv"
    )
    df = pd.read_csv(data_path)

    # Ensure DATE is treated as year (int)
    df["DATE"] = df["DATE"].astype(int)

    # Standardize column names we will rely on
    # (If any of these are missing, we simply skip those features later.)
    return df


@st.cache_data
def compute_station_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple linear trends per station for:
      - Average annual temperature (TAVG)
      - Number of very hot days (>= 90°F), if available
    using a least-squares line fit (slope only).
    """
    rows = []
    for name, g in df.groupby("NAME"):
        years = g["DATE"].values

        if len(np.unique(years)) < 3:
            # Too few points for a meaningful trend
            continue

        # Temperature trend
        if "TAVG" in g.columns:
            tavg = g["TAVG"].values
            slope_tavg = np.polyfit(years, tavg, 1)[0]
        else:
            slope_tavg = np.nan

        # Hot days trend (>= 90°F), if we engineered that field
        if "heat_extreme_days" in g.columns:
            dx90 = g["heat_extreme_days"].values
            slope_dx90 = np.polyfit(years, dx90, 1)[0]
        else:
            slope_dx90 = np.nan

        rows.append(
            {
                "NAME": name,
                "tavg_trend_per_year": slope_tavg,
                "dx90_trend_per_year": slope_dx90,
            }
        )

    trends = pd.DataFrame(rows)
    return trends


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
df = load_data()
trends_df = compute_station_trends(df)

min_year = int(df["DATE"].min())
max_year = int(df["DATE"].max())

# Some convenience lists for dropdowns
station_names = sorted(df["NAME"].unique())

metric_options = {
    "Average temperature (TAVG, °F)": "TAVG",
    "Total precipitation (PRCP, inches)": "PRCP",
    "Total snowfall (SNOW, inches)": "SNOW",
}

extreme_options = {
    "Very hot days ≥ 90°F": "heat_extreme_days",
    "Very cold days ≤ 32°F": "cold_extreme_days",
}

# ---------------------------------------------------------------------
# Sidebar – narrative + controls
# ---------------------------------------------------------------------
st.sidebar.title("NOAA NYC Climate Explorer")

st.sidebar.markdown(
    """
This dashboard summarizes **annual climate conditions and trends** for weather
stations in the **New York City metropolitan area** using a subset of the
NOAA GSOM annual summaries (2020–2025).

Use the controls below to focus on specific stations, years, and metrics.
"""
)

year_range = st.sidebar.slider(
    "Select year range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
)

selected_metric_label = st.sidebar.selectbox(
    "Primary climate metric for comparisons",
    list(metric_options.keys()),
)
selected_metric = metric_options[selected_metric_label]

selected_stations = st.sidebar.multiselect(
    "Stations for detailed exploration",
    station_names,
    default=[
        s
        for s in station_names
        if "CENTRAL PARK" in s or "JFK" in s or "NEWARK" in s
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**How to read this app**

- Start with the **Regional overview** tab to see which stations are
  systematically warmer or wetter than others.
- Use **Station explorer** to drill into any location's annual time series.
- Open **Trends & extremes** to see where warming and very hot days are
  increasing fastest.
- The **Methods & notes** tab documents the dataset, basic processing steps,
  and interpretation cautions.
"""
)

# Filter the main dataframe by selected year range
mask_year = (df["DATE"] >= year_range[0]) & (df["DATE"] <= year_range[1])
df_year = df.loc[mask_year].copy()

# ---------------------------------------------------------------------
# Main title & description
# ---------------------------------------------------------------------
st.title("02 – Station Climate Profiles and Trends (NOAA NYC 2020–2025)")

st.markdown(
    """
This application turns the NOAA GSOM annual summaries for the NYC metro area
into **station-level climate profiles and trend diagnostics**. The goal is not
only to show interactive plots, but to tell a clear story about:

- How conditions vary **across stations** (who is warmest, wettest, snowiest)
- How conditions change **over time** (short-term trends in temperature and extremes)
- What this implies for **urban climate risk** and **infrastructure planning**

Use the tabs below to move from high-level comparisons to station-level detail.
"""
)

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab_overview, tab_station, tab_trends, tab_methods = st.tabs(
    ["Regional overview", "Station explorer", "Trends & extremes", "Methods & notes"]
)

# ---------------------------------------------------------------------
# TAB 1 – Regional overview
# ---------------------------------------------------------------------
with tab_overview:
    st.subheader("Regional climate ranking by station")

    st.markdown(
        f"""
In this section we collapse the chosen period **{year_range[0]}–{year_range[1]}**
into summary metrics by station. For the selected climate variable:

- We compute the **mean** value over the chosen years for each station.
- We then sort stations from highest to lowest and show the top locations.

This gives a quick sense of which parts of the metro area are systematically
warmer, wetter, or snowier than others.
"""
    )

    if selected_metric not in df_year.columns:
        st.warning(
            f"The column `{selected_metric}` is not present in the dataset. "
            "Try a different metric."
        )
    else:
        agg_func = "mean" if selected_metric == "TAVG" else "sum"
        station_profiles = (
            df_year.groupby(["NAME"], as_index=False)[selected_metric]
            .agg(agg_func)
            .rename(columns={selected_metric: "value"})
        )

        top_n = st.slider(
            "Number of stations to display", 5, min(25, len(station_profiles)), 15
        )

        top_profiles = station_profiles.sort_values("value", ascending=False).head(
            top_n
        )

        fig_bar = px.bar(
            top_profiles.sort_values("value"),
            x="value",
            y="NAME",
            orientation="h",
            labels={"value": selected_metric_label, "NAME": "Station"},
            title=f"Top {top_n} stations by {selected_metric_label.lower()} "
            f"({year_range[0]}–{year_range[1]})",
        )

        fig_bar.update_layout(height=600, margin=dict(l=220, r=40, t=60, b=40))
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown(
            """
**Interpretation tips**

- Airports and coastal sites often look different from inland suburban stations.
- For precipitation and snowfall, single outliers may reflect local convective storms.
- Because we are working with only a few years (2020–2025), treat rankings as
  **indicative**, not definitive climatology.
"""
        )

# ---------------------------------------------------------------------
# TAB 2 – Station explorer
# ---------------------------------------------------------------------
with tab_station:
    st.subheader("Station explorer")

    st.markdown(
        """
This view focuses on one or more stations and shows their **annual time series**
for temperature and precipitation. Use it to compare how different locations in
the metro area experienced the same period.
"""
    )

    if not selected_stations:
        st.info("Select at least one station in the sidebar to see details.")
    else:
        df_sel = df[df["NAME"].isin(selected_stations)].copy()
        df_sel = df_sel[(df_sel["DATE"] >= year_range[0]) & (df_sel["DATE"] <= year_range[1])]

        # Temperature time series
        if "TAVG" in df_sel.columns:
            fig_t = px.line(
                df_sel,
                x="DATE",
                y="TAVG",
                color="NAME",
                markers=True,
                labels={"DATE": "Year", "TAVG": "Average temperature (°F)", "NAME": "Station"},
                title="Average annual temperature by station",
            )
            fig_t.update_layout(height=450)
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.warning("Column `TAVG` is not available in this dataset.")

        # Precipitation time series
        if "PRCP" in df_sel.columns:
            fig_p = px.line(
                df_sel,
                x="DATE",
                y="PRCP",
                color="NAME",
                markers=True,
                labels={"DATE": "Year", "PRCP": "Total precipitation (inches)", "NAME": "Station"},
                title="Total annual precipitation by station",
            )
            fig_p.update_layout(height=450)
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.warning("Column `PRCP` is not available in this dataset.")

        st.markdown(
            """
**What to look for**

- Do core NYC stations (e.g., Central Park, JFK, Newark) track each other closely,
  or do they diverge in certain years?
- Are inland stations consistently cooler than coastal ones?
- Do some stations show suspicious jumps that might indicate instrumentation or
  siting changes rather than real climate signals?
"""
        )

# ---------------------------------------------------------------------
# TAB 3 – Trends & extremes
# ---------------------------------------------------------------------
with tab_trends:
    st.subheader("Short-term warming and hot-day trends")

    st.markdown(
        """
Here we estimate **simple linear trends** in:

- Average annual temperature (**TAVG**, °F/year)
- Number of **very hot days (≥ 90°F)** per year, where that derived variable is available

Trends are calculated separately for each station using a least-squares line fit
over the selected year range. With only 6 years of data (2020–2025), these
should be interpreted as **short-term signals**, not long-term climate normals.
"""
    )

    # Recompute trends on the filtered year range so the slider actually matters
    trends_year = compute_station_trends(df_year)

    # Temperature trend bar chart
    if not trends_year.empty and "tavg_trend_per_year" in trends_year.columns:
        top_n_trend = st.slider(
            "Number of stations to show in trend charts", 5, min(25, len(trends_year)), 15
        )

        # Warmest trend (largest positive slope)
        warmers = trends_year.sort_values("tavg_trend_per_year", ascending=False).head(
            top_n_trend
        )

        fig_trend_t = px.bar(
            warmers.sort_values("tavg_trend_per_year"),
            x="tavg_trend_per_year",
            y="NAME",
            orientation="h",
            labels={
                "tavg_trend_per_year": "Slope (°F per year)",
                "NAME": "Station",
            },
            title=f"Stations with strongest warming trend in TAVG ({year_range[0]}–{year_range[1]})",
        )
        fig_trend_t.update_layout(height=600, margin=dict(l=220, r=40, t=60, b=40))
        st.plotly_chart(fig_trend_t, use_container_width=True)
    else:
        st.info("Not enough data to compute temperature trends for this period.")

    # Hot-day trend bar chart
    if "dx90_trend_per_year" in trends_year.columns and not trends_year["dx90_trend_per_year"].isna().all():
        st.markdown("---")
        st.markdown("### Change in very hot days (≥ 90°F) per year")

        hottest = trends_year.sort_values(
            "dx90_trend_per_year", ascending=False
        ).head(top_n_trend)

        fig_trend_dx = px.bar(
            hottest.sort_values("dx90_trend_per_year"),
            x="dx90_trend_per_year",
            y="NAME",
            orientation="h",
            labels={
                "dx90_trend_per_year": "Slope (days ≥90°F per year)",
                "NAME": "Station",
            },
            title="Stations with strongest increase in very hot days",
        )
        fig_trend_dx.update_layout(height=600, margin=dict(l=220, r=40, t=60, b=40))
        st.plotly_chart(fig_trend_dx, use_container_width=True)
    else:
        st.info(
            "Derived field `heat_extreme_days` was not found or does not contain enough data "
            "to compute trends in hot days."
        )

    st.markdown(
        """
**How to read these charts**

- Positive slopes indicate **warming** or **more hot days** over the period.
- Negative slopes indicate cooling or fewer hot days (often not statistically
  meaningful over such a short window).
- Compare stations rather than obsessing over the exact numerical value of the slope.
"""
    )

# ---------------------------------------------------------------------
# TAB 4 – Methods & notes
# ---------------------------------------------------------------------
with tab_methods:
    st.subheader("Methods, caveats, and extensions")

    st.markdown(
        f"""
### Dataset

- Source: NOAA Global Summary of the Year (GSOM/GSOD Annual Summary)
- Region: New York City metropolitan area
- Period: **{min_year}–{max_year}**
- Spatial unit: individual weather stations (NAME / STATION)

The cleaned dataset used in this app is stored as
`data/processed/noaa_nyc_annual_clean.csv` in the GitHub repository.
"""
    )

    st.markdown(
        """
### Key variables

- **TAVG** – Average temperature over the year (°F)
- **TMAX / TMIN** – Mean of daily maxima / minima (°F)
- **PRCP** – Total annual precipitation (inches)
- **SNOW** – Total annual snowfall (inches)
- **heat_extreme_days** – Engineered: count of days with TMAX ≥ 90°F (if available)
- **cold_extreme_days** – Engineered: count of days with very cold conditions (if available)

Additional engineered indices (degree-days, wetness index, etc.) can be layered
into future versions of the dashboard.
"""
    )

    st.markdown(
        """
### Trend estimation

For each station, trends are estimated with a simple **least-squares linear fit**:

- Slope of TAVG vs. year → °F change per year
- Slope of `heat_extreme_days` vs. year → change in number of very hot days per year

This is intentionally lightweight and transparent, but it has limitations:

- Only **six years** of data (2020–2025) → results are sensitive to individual years
- No adjustment for autocorrelation, step changes, or inhomogeneities
- No spatial smoothing across stations

The goal is to provide **directional insight**, not formal climate attribution.
"""
    )

    st.markdown(
        """
### Possible extensions

If this project is extended for a more advanced portfolio piece, next steps
could include:

- Adding **degree-day** and **wetness index** visualizations for energy/water planning
- Incorporating **station metadata** (urban vs. rural, elevation, land cover)
- Exporting a formal **technical report** with all figures and summary tables
- Connecting the app to a **larger historical window** (e.g., 1980–present) for robust trends

For now, the dashboard demonstrates the ability to go from raw NOAA data to a
**fully documented, interactive climate analytics product** suitable for
stakeholders and hiring managers.
"""
    )
