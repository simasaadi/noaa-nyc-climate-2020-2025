NOAA NYC Climate Explorer (2020–2025)

Short-term climate analysis of NOAA GSOM annual summaries for NYC-area weather stations. The project focuses on temperature, precipitation, and snowfall variability across the region and delivers a full interactive dashboard built with Streamlit and Plotly.

🚀 Live Dashboard


Click to open the interactive NYC climate dashboard.

Overview

This repository includes a clean workflow from data preparation to visualization:

Processed NOAA GSOM data (2020–2025)

Jupyter notebooks for EDA, profiling, and short-term trend estimation

A Streamlit dashboard with metric filters, station rankings, time-series views, and a geo-spatial map

Plotly-based visuals for interactive exploration

This project is structured as a focused, end-to-end analytics workflow that works well for both research and portfolio presentation.

Repository Structure
noaa-nyc-climate-2020-2025/
│
├── data/
│   ├── processed/
│   └── raw/ (optional)
│
├── notebooks/
│   ├── 01_eda_noaa_nyc.ipynb
│   └── 02_station_profiles_and_trends.ipynb
│
├── app/
│   ├── app.py
│   ├── requirements.txt
│
└── README.md

Data & Methods

Data Source: NOAA Global Summary of the Month (GSOM).
Coverage: NY + NJ stations with complete 2020–2025 records.

Methods used:

Data cleaning and validation

Multi-year station profiles

Simple linear trend estimation (temperature, precipitation, snowfall)

Spatial comparison using station coordinates

Interactive dashboard development with Streamlit + Plotly

Run Locally
pip install -r app/requirements.txt
streamlit run app/app.py
