# Movie Poster Data Analysis Project 3

**DS 4002: NBA All Stars** **Authors:** Nick Larson, Bowen Slingluff, Andrew Patterson  
**Date:** April 15, 2026  
**University of Virginia**

## Executive Summary
This document outlines the data collection and analytical approach for a study evaluating whether a computer model can accurately guess a movie’s genre, ratings, and box office performance solely based on its poster image. By converting poster images into numerical features and combining them with structured movie and economic data, we aim to evaluate whether visual marketing and economic factors together can explain a movie's financial and critical success.

---

## Goals & Hypothesis

* **Goal Statement:** Classify movie attributes such as genre and box office results based on poster images.
* **Quantifiable Goal:** Combine movie poster image data with inflation, movie ratings, and box office revenue data. Apply machine learning models to classify genres and predict financial performance, and evaluate how well visual features and economic conditions together explain the variation in a movie's success.
* **Hypothesis:** We hypothesize that a computer vision model will be able to classify genre with high accuracy (greater than 80%) but will struggle with box office results and ratings. We do not expect movie posters to provide significant contextual evidence toward the revenue or ratings of a film.
* **Research Question:** To what degree can computer vision models accurately classify movie genres, ratings, and box office performance from poster images?

---

## Dataset Establishment

Our dataset consists of movie metadata and poster image data from The Movie Database (TMDB), downloaded from a pre-made Hugging Face dataset. **Instead of downloading the massive 42-gigabyte raw dataset all at once, we utilize Hugging Face's streaming API to dynamically pull data.** We map low-level visual features (colors, textures, composition) to high-level metadata using the `id` as the primary key. **To ensure data quality, we apply strict filtering during the collection phase, dropping any movies that lack genre tags or report $0 in box office revenue, ultimately capping our dataset at a representative sample of 10,000 movies.** Furthermore, release date serves as a control variable to normalize shifting visual aesthetic trends over time.

### Data Dictionary

| Column Name | Description | Example |
| :--- | :--- | :--- |
| `id` | The unique numerical identifier from TMDB. | `346698` |
| `title` | The official display name of the movie. | `Barbie` |
| `genres` | A JSON List of objects representing the movie's categories with genre ID and name. | `[{"id": 35, "name": "Comedy"}]` |
| `release_date` | The date the movie was first released in theaters (YYYY-MM-DD). | `2023-07-19` |
| **`release_year`** | **The parsed year of the movie's release, used for CPI lookups.** | **`2023`** |
| `revenue` | The total worldwide box office gross in USD (primary regression target). | `1434628000` |
| **`revenue_adj`** | **The inflation-adjusted worldwide box office gross (Base Year: 2026), which is log-transformed during modeling.** | **`6.110814e+08`** |
| `vote_average` | A 0–10 scale representing the average rating given by TMDB users. | `6.56` |
| `vote_count` | The total number of TMDB users who have submitted a rating. | `10937` |
| `image` | The raw pixel data or file path to the movie poster. | *(Image Path/Data)* |
| `image_width` | The horizontal resolution of the poster image in pixels. | `500` |
| `image_height` | The vertical resolution of the poster image in pixels. | `750` |
| `cpi_date` | The year and month of the CPI observation, formatted to match release date. | `1974-01` |
| `cpi_value` | The CPI Price Index (Base: 1982-84=100), used to calculate inflation-adjusted revenue. | `46.6` |

---

## Methodology & Analysis Plan

### 1. Workflow
**Data Collection** → **Image Preprocessing** → **Feature Extraction** → **Merge with Movie/Economic Data** → **Train ML Model** → **Evaluate Model Performance** → **Interpret Results**

### 2. Preprocessing
All data wrangling and image processing will be conducted using the GPU nodes on Rivanna. 
* **Merge & Normalize:** Merge the image dataset with secondary financial and TMDB ratings datasets using the unique TMDB ID. Revenue is adjusted for inflation utilizing Consumer Price Index (CPI) data based on each film’s release **year, benchmarked to 2026 dollars**. **Because revenue distributions are highly skewed, the adjusted revenue is also log-transformed (`np.log1p`) before being fed into the regression model.**
* **Image Standardization:** All movie posters will be standardized to a uniform resolution (224x224 pixels).
* **Data Cleaning:** The target variables will be cleaned, genre lists will be one-hot encoded for multi-label classification, continuous variables scaled, and any records with missing images, corrupted data, or insufficient vote counts will be dropped.

### 3. Modeling Approach
This project leverages **transfer learning** using the **ResNet-50** architecture. By using this pre-trained, deep convolutional neural network as a foundation, we bypass training from scratch and utilize its locked-in foundational ability to recognize complex visual features (textures, shapes, compositions).
* **Branch 1 (Multi-label Classification):** Fine-tuned to identify the specific stylistic markers of film genres.
* **Branch 2 (Dual-target Regression):** Fine-tuned to predict continuous outcomes: inflation-adjusted revenue and `vote_average`. 

### 4. Evaluation Strategy
* **Genre Classification:** Primary metric is **Classification Accuracy** (targeting >80%), supplemented by **F1-scores** to account for genre imbalances.
* **Revenue and Ratings Predictions:** Evaluated using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**. These metrics will be benchmarked against a **"dummy" baseline model** (which guesses the historical average revenue/rating for every movie) to definitively measure if visual elements hold predictive power over financial and critical success.

---

## Software and Platform

**Environment:** Jupyter Notebook / Google Colab / Rivanna (UVA HPC)  
**OS:** MacOS, Linux (HPC)  

**Required Python Packages:**
*(Install before running via `pip install pandas numpy matplotlib statsmodels scipy seaborn scikit-learn datasets huggingface_hub ratelimit tqdm pillow`)*
* `pandas`
* `numpy`
* `matplotlib` & `seaborn`
* `statsmodels` & `scipy`
* `scikit-learn`
* `datasets` & `huggingface_hub`
* `pillow`
* *Note: Deep learning libraries (e.g., PyTorch or TensorFlow) are required for ResNet-50 processing.*

---
## Repository Contents & File Map
This repository contains all the code, raw data references, and output metrics required to replicate our poster analysis study. Below is a map of the repository's folder and file structure:

```text
├── DATA/                     # Folder for raw and pre-processed data
│   ├── movie_data_for_eda.csv    # Initial movie metadata
│   └── Original CPI-U data.csv   # Consumer Price Index data
│   └── data_access_instructions.md # How to get data
│   └── Data Appendix P3.pdf      # Data appendix
├── OUTPUT/                   # Generated evaluation metrics and figures
│   └── eda_plots.png             # Initial eda plots
│   └── Filtered Revenue Performance.png
│   └── Genre Classification Accuracy.png
│   └── Multi-Task Combined Loss.png
│   └── Revenue Prediction Error.png
├── SCRIPTS/                  # Dynamically generated folder for image downloads
│   └── plots.ipynb               # Main Jupyter notebook for exploratory data analysis
│   └── analysis.ipynb      # Notebook containing the ResNet-50 modeling
└── README.md                 # Project orientation and reproduction instructions
```
---
## Instructions to Reproduce Results

1.  **Download Local Datasets:** Download `movie_data_for_eda.csv` and the original CPI-U dataset, and place both files directly into the `DATA` folder.
2.  **API and Cloud Data Setup:** You will need an active TMDB API key to fetch the supplementary movie metadata. The **poster image dataset is streamed dynamically from Hugging Face directly within the code, which evaluates and downloads exactly 10,000 valid images to a local `./posters` directory (~491MB footprint)**, so no manual downloading of the 42GB raw file is required.
3.  **Environment Setup:** Due to the massive size of the image dataset, local processing is not feasible. You must connect to a high-performance computing environment (such as UVA's Rivanna HPC) and allocate GPU nodes. Open `analysis.ipynb` (or the respective modeling notebook) within this environment. 
4.  **Run Cells:** Run all notebook cells from top to bottom.
5.  **Processing:** The notebook will automatically **stream** the Hugging Face images and TMDB API data, clean the merged dataset **(filtering out $0 revenue entries)**, resize all posters to 224x224 pixels, adjust revenue for CPI inflation **(using the 2026 benchmark)**, and train the ResNet-50 models.
6.  **Outputs:** All evaluation metrics, F1-scores, error comparisons (MAE/RMSE) against the dummy baseline, and generated figures will appear in the `OUTPUT` folder.

---

## References

1. "Movie Posters 100k ControlNet," *Hugging Face Datasets*, https://huggingface.co/datasets/stzhao/movie_posters_100k_controlnet (accessed Apr. 1, 2026).
2. "Consumer Price Index (CPI) Databases," *U.S. Bureau of Labor Statistics*, https://www.bls.gov/data/home.htm#prices (accessed Apr. 1, 2026).
3. "Movie Dataset Financials," *Google Sheets*, https://docs.google.com/spreadsheets/d/1bYVyxLtbq9bBDxdLtSXQOpZlqNKHLGtO/ (accessed Apr. 1, 2026).
4. "Getting Started," *TMDB Developer Documentation*, https://developer.themoviedb.org/docs/getting-started (accessed Apr. 1, 2026).
5. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770-778. [Online]. Available: https://arxiv.org/abs/1512.03385 (accessed Apr. 1, 2026).

---

## License

This project uses the **MIT License**.
