# Data Access and Reproduction Instructions

**Project:** Movie Poster Data Analysis (ResNet-50 CV Study)  
**Purpose:** This document outlines the required steps to obtain and construct the primary dataset used for this project. Because the raw image data and associated weights exceed standard GitHub storage limits (approx. 500 MB for the image sample alone), the dataset must be pulled dynamically via API and stored locally or on a high-performance computing cluster.

### Prerequisites and Environment Requirements
To execute the data acquisition scripts, you will need the following:
1. **The TMDB API Key:** Register for a developer account at The Movie Database (TMDB) to obtain an API key.
2. **Hugging Face Account & Token:** Create an account on Hugging Face and generate an access token to stream the image dataset.
3. **Computing Environment:** Due to the volume of images and subsequent deep learning requirements, it is highly recommended to run this data pipeline on a high-performance computing environment (such as UVA's Rivanna cluster) with adequate storage allocation.

### Step 1: Environment Setup
Ensure your environment has the required Python libraries installed, specifically for handling APIs and image streaming:
* `pandas`, `numpy`, `requests`, `datasets`, `huggingface_hub`, `Pillow`, `ratelimit`, and `tqdm`.

You must export your API keys as environment variables before running the data collection scripts:
`export HF_TOKEN="your_huggingface_token"`
`export TMDB_API_KEY="your_tmdb_api_key"`

### Step 2: Sourcing Baseline Inflation Data
Before pulling the API data, ensure the local CPI dataset is in your working directory. 
* **File:** `Original CPI-U data.csv`
* **Function:** This file contains historical Consumer Price Index (CPI) data used to adjust all historical box office revenues to a 2026 baseline. The script automatically parses this table to build a lookup dictionary for the inflation-adjustment logic.

### Step 3: Fetching Images and Metadata
The raw image data is sourced from the Hugging Face repository `stzhao/movie_posters_100k_controlnet`. Because the full dataset is 42+ gigabytes, the script utilizes the `datasets` library with `streaming=True` to process entries one by one.

**Data Filtering Protocol:**
The collection script iterates through the Hugging Face stream and pings the TMDB API for each movie ID. A movie is only saved to the final dataset if it passes the following strict criteria:
1. **Valid Genres:** The movie must have at least one defined genre.
2. **Valid Revenue:** The movie must report a box office revenue strictly greater than $0.

If the movie passes these filters, the script:
* Saves the `.jpg` poster to a local `./posters` directory.
* Appends the TMDB metadata (id, genres, raw revenue, vote average) to a list.
* Automatically terminates once the target sample size (10,000 viable movies) is reached.

### Step 4: Finalizing the Dataset
After the image collection is complete, the script executes a secondary pass to fetch the release year for each movie. Using the release year and the `Original CPI-U data.csv` lookup table, the script generates a final `revenue_adj` column, standardizing the financial target variable.

**Expected Output:**
Successful execution of the data pipeline will yield two primary artifacts in your Data folder:
1. `metadata_final.csv`: A lightweight dataset (~775 KB) containing the IDs, genres, and adjusted revenues for the sample.
2. `./posters/`: A directory containing the corresponding `.jpg` files for the analysis (~500 MB).
