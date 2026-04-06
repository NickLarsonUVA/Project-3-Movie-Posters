Project-2-Bitcoin

Bitcoin vs Inflation Time Series Analysis (DS 4002 Project)

Project Overview

This project investigates whether Bitcoin behaves as an inflation-sensitive asset by analyzing the relationship between Bitcoin returns, CPI inflation, and inflation expectations. Using a Vector Autoregression or VAR model, we examine whether inflation shocks produce measurable short term or long term effects on Bitcoin prices.

Research Question

Does Bitcoin respond positively to increases in inflation and inflation expectations, and are these responses short-term or persistent over time?

Modeling Approach

We use a Vector Autoregression (VAR) time series model including:

Bitcoin returns
CPI inflation (year-over-year change)
5-Year Breakeven Inflation Rate
S&P 500 returns (control variable) Impulse Response Functions were used to analyze how Bitcoin responds to inflation shocks over time.
Repository Contents

Project-2-Bitcoin/

README.md LICENSE.md

DATA/ CPIAUCSL.csv T5YIE.csv btc-usd-max.csv master_dataset_daily.csv master_dataset_monthly.csv

SCRIPTS/ eda.ipynb

OUTPUT/ growth_chart.png

Section 1: Software and Platform

Software Used:

Jupyter Notebook / Google Colab
Python Packages Required Install before running: -pandas -numpy -matplotlib -statsmodels -scipy -seaborn Install with: -pip install pandas numpy matplotlib statsmodels scipy seaborn
Platform developed on:

MacOS Google Colab environment

Section 2: Data Sources

Data used in this project comes from:

Bitcoin historical data (CoinGecko)
CPI data (Bureau of Labor Statistics)
5-Year Breakeven Inflation Rate (FRED)
S&P 500 data (FRED) All datasets were merged into a monthly time series dataset.
Section 3: Instructions to Reproduce Results

Step 1: Download the datasets located in the DATA folder. Step 2: Open eda.ipynb in Jupyter Notebook or Google Colab. Step 3: Run all notebook cells from top to bottom. Step 4: The notebook will clean the data and generate the analysis. Step 5: All figures, tables, and model outputs will appear in the OUTPUT folder. Step 6: Review the OUTPUT folder for summary statistics, figures, and VAR model results.

Analysis Plan Summary

Our workflow follows: Data Collection → Data Cleaning → Stationarity Testing → VAR Model → Impulse Response Analysis → Evaluation Evaluation criteria include: Statistical significance of relationships Model stability Evidence of Bitcoin response to inflation shocks

Quantifiable Goal

This project will be considered successful to us if we can determine whether Bitcoin responds significantly to inflation changes and whether those effects are temporary or persistent.

References

[1] CoinGecko Bitcoin Historical Data [2] Bureau of Labor Statistics CPI Data [3] Federal Reserve Economic Data (FRED) [4] MathWorks VAR Model Documentation [5] Choi & Shin (2022) Bitcoin Inflation Hedge Study

Authors

DS 4002 NBA All Stars Group: Bowen Slingluff Nick Larson Andrew Patterson University of Virginia DS 4002 – Data Science Project Lifecycle Spring 2026

License

This project uses the MIT License.
