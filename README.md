# noaa-nyc-climate-2020-2025
Exploratory climate analysis of NOAA GSOM annual summaries for the NYC metro area (2020–2025).

NOAA NYC Climate Explorer (2020–2025)
This project analyzes short-term climate conditions and trends across weather stations in the New York City metropolitan area using annual summaries from the NOAA GSOM dataset (2020–2025).
🚀 Live Dashboard

Click above to open the fully interactive NYC climate dashboard.

It combines a clean end-to-end workflow:

Reproducible data cleaning

Exploratory analysis notebooks

Station-level climate profiling

Trend estimation

A full interactive Streamlit dashboard

Clear documentation

The result is a compact but professional climate-intelligence tool that helps identify hot spots, wet spots, outliers, and short-term warming signals across the metro area.

Project Overview

This repository contains:

Processed NOAA climate data for all stations with valid 2020–2025 annual summaries

A series of analysis notebooks that build station profiles, compute metrics, and derive trends

A fully interactive Streamlit dashboard (published online)

A clean project structure that mirrors typical analytics or research workflows

The project can serve as:

A portfolio piece demonstrating strong data-analysis, visualization, and dashboarding skills

A climate-monitoring tool for researchers, planners, or data-driven teams

A foundation for longer-term climate-risk analysis

Live Dashboard

Explore the interactive app:

➡ NOAA NYC Climate Explorer (Streamlit App)

Link: https://noaa-nyc-climate-2020-2025.streamlit.app/
(Replace with your actual app link)

Features of the dashboard:

Year-range filtering (2020–2025)

Multiple climate metrics (temperature, precipitation, snowfall, extremes)

Station rankings with heat-map style encoding

Time-series comparisons vs metro mean

Station-level trend estimation

Interactive map view showing spatial climate patterns
Repository Structure
noaa-nyc-climate-2020-2025/
│
├── data/
│   ├── raw/                       # Original downloaded NOAA data (optional)
│   ├── processed/
│   │   └── noaa_nyc_annual_clean.csv
│   └── noaa_nyc_annual_2020_2025.csv
│
├── notebooks/
│   ├── 01_eda_noaa_nyc.ipynb      # Initial EDA and data checks
│   └── 02_station_profiles_and_trends.ipynb
│
├── app/
│   ├── app.py                     # Streamlit dashboard
│   ├── requirements.txt           # Deployment dependencies
│
├── src/                           # Utility scripts (optional)
│
├── README.md
└── LICENSE (optional)
Data Source

All data in this project comes from the
NOAA Global Summary of the Month (GSOM) dataset.

Dataset used:

NYC metro stations (NY + NJ)

2020–2025 annual summaries

Variables include:

Average temperature (°F)

Total precipitation (inches)

Total snowfall (inches)

Heat/cold extreme days (if present)

Lat/long, station IDs, station names

This project does not modify NOAA data beyond:

Cleaning invalid records

Aligning column names

Aggregating annual metrics

Methodology
1. Data Cleaning

Performed in 01_eda_noaa_nyc.ipynb:

Validate NOAA fields and units

Harmonize column names across stations

Remove corrupted/incomplete rows

Filter to stations with full 2020–2025 coverage

Export a clean annual dataset

2. Station Climate Profiles

In 02_station_profiles_and_trends.ipynb:

Compute multi-year averages for:

Temperature

Precipitation

Snowfall

Extreme heat/cold days (if available)

Rank stations relative to the metro average

Compute differences (e.g., hotter/cooler, wetter/drier)

3. Trend Estimation
4. A simple linear regression is applied for each station:

y = m*x + b
slope = m
Interpreted as:

°F per year (for temperature)

inches per year (precipitation/snowfall)

With only 6 years of data, these are treated as short-term signals, not long-term climate trends.

4. Dashboard Development

The Streamlit app integrates:

Multi-metric control panel

Station ranking visualizations (heatmap bars)

Year-to-year time series

Trend distribution histograms

A map visualizing geographic patterns

All plots use Plotly for interactivity.

Key Findings (Example – adjust after reviewing your dashboard)

Several NYC metro stations show persistent warm anomalies above the regional mean.

A small cluster of stations in central NJ demonstrates slightly stronger short-term warming signals.

Precipitation varies widely across the region, with coastal stations often wetter than inland ones.

Year-to-year temperature variability is highly synchronized across stations, suggesting strong regional forcing.

Running the Project Locally
1. Clone the repository
git clone https://github.com/simasaadi/noaa-nyc-climate-2020-2025.git
cd noaa-nyc-climate-2020-2025

2. Create environment + install dependencies
pip install -r app/requirements.txt

3. Run the Streamlit app
streamlit run app/app.py


The dashboard will open in your browser.

Skills Demonstrated

This project highlights expertise in:

Data cleaning and preprocessing

EDA for environmental datasets

Time-series analysis

Trend modelling and regression

Interactive dashboards (Streamlit + Plotly)

Multi-station spatial analysis

Project structuring and documentation

Communicating climate insights to technical and non-technical audiences

Future Work

Potential extensions:

Incorporate long-term NOAA normals (30-year)

Add seasonal breakdowns

Integrate air quality or humidity data

Apply clustering to group similar station profiles

Automate data refresh using NOAA API
