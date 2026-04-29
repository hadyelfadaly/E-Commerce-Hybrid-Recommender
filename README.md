# E-Commerce Recommender System

This project implements a hybrid recommender system for an e-commerce platform, combining association rule mining (Apriori/FP-Growth) with content-based filtering (TF-IDF). The goal is to provide personalized product recommendations to users based on their purchase history and product features.

## Steps to Implement (in progress)

1. Load and clean the dataset (Done by Hady)
2. Run Apriori/FP-Growth to extract rules (Done by Hady)
3. Build TF-IDF vectors for products (Done by Yassin)
4. For a given user, get candidates from both methods (Done by Yassin)
5. Score and rank them using the hybrid formula (Done by Yassin)
6. Evaluate with Precision/Recall (Done by Hady)
7. Write the report 

## Dataset

The dataset used in this project is the [Online Retail Dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii) from the UCI Machine Learning Repository. It contains transactional data for a UK-based online retail store, including product descriptions, quantities, and customer information.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/hadyelfadaly/E-Commerce-Hybrid-Recommender.git
    ```
2. Navigate to the Code directory:
   ```bash
   cd Code
   ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
