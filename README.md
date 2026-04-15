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

Our dataset consists of movie metadata and poster image data from The Movie Database (TMDB), downloaded from a pre-made Hugging Face dataset. Because of the massive 42-gigabyte footprint of our raw poster data, we will load this data into our scripts programmatically using the University of Virginia's high-performance computer, Rivanna. 

We map low-level visual features (colors, textures, composition) to high-level metadata using the `id` as the primary key. A minimum threshold on `vote_count` is applied to prevent movies with very few ratings from skewing the results. Furthermore, release date serves as a control variable to normalize shifting visual aesthetic trends over time.

### Data Dictionary

| Column Name | Description | Example |
| :--- | :--- | :--- |
| `id` | The unique numerical identifier from TMDB. | `346698` |
| `title` | The official display name of the movie. | `Barbie` |
| `genres` | A JSON List of objects representing the movie's categories with genre ID and name. | `[{"id": 35, "name": "Comedy"}]` |
| `release_date` | The date the movie was first released in theaters (YYYY-MM-DD). | `2023-07-19` |
| `revenue` | The total worldwide box office gross in USD (primary regression target). | `1434628000` |
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
* **Merge & Normalize:** Merge the image dataset with secondary financial and TMDB ratings datasets using the unique TMDB ID. Revenue is adjusted for inflation utilizing Consumer Price Index (CPI) data based on each film’s release date.
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
*(Install before running via `pip install pandas numpy matplotlib statsmodels scipy seaborn`)*
* `pandas`
* `numpy`
* `matplotlib` & `seaborn`
* `statsmodels` & `scipy`
* *Note: Deep learning libraries (e.g., PyTorch or TensorFlow) are required for ResNet-50 processing.*

---

## Instructions to Reproduce Results

1.  **Download the Datasets:** Ensure data from Hugging Face, BLS (CPI), and TMDB are located in the `DATA` folder.
2.  **Environment Setup:** Open `eda.ipynb` (or the respective modeling notebook) in Jupyter Notebook or your preferred IDE. Connect to Rivanna if processing the full 42GB dataset.
3.  **Run Cells:** Run all notebook cells from top to bottom.
4.  **Processing:** The notebook will clean the data, resize images to 224x224, adjust for CPI inflation, and train the ResNet-50 models.
5.  **Outputs:** All evaluation metrics, F1-scores, error comparisons (MAE/RMSE) against the dummy baseline, and generated figures will appear in the `OUTPUT` folder.

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
