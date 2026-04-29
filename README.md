# E-Commerce Hybrid Recommender System

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![mlxtend](https://img.shields.io/badge/mlxtend-FP--Growth-3a3a3a?style=flat-square)

A hybrid product recommendation engine combining **FP-Growth Association Rule Mining** with **TF-IDF Content-Based Filtering**, built on the Online Retail II dataset (~1M transactions). The system generates personalized recommendations per customer and evaluates them using Precision@5, Recall@5, and F1@5.

---

## Architecture

```
Raw Transactions (Online Retail II)
        │
        ▼
   Preprocessing
   (clean, merge, split)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
 interactions.csv                    metadata.csv
 (Invoice × StockCode)           (StockCode, Description)
        │                                  │
        ▼                                  ▼
  FP-Growth Mining                  TF-IDF Vectorizer
  (basket matrix)               (cosine similarity matrix)
        │                                  │
        ▼                                  ▼
  AR Candidates                   Content Candidates
  (confidence score)              (cosine score)
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
          Hybrid Scoring Formula
     α × AR_score + (1−α) × content_score
                       │
                       ▼
            Top-K Recommendations
```

---

## Dataset

**Online Retail II** — UCI Machine Learning Repository  
UK-based online retailer, December 2009 – December 2011

| Attribute | Value |
|---|---|
| Raw rows | ~1,067,371 |
| After cleaning | ~771,542 |
| Unique customers | ~5,878 |
| Unique products | ~4,070 StockCodes |
| Unique invoices | ~49,000+ |

**Preprocessing steps:**
- Merged both annual Excel sheets
- Dropped rows with `Quantity ≤ 0` or `Price ≤ 0` (returns/cancellations)
- Dropped rows with missing `Customer ID`
- Capped `Quantity` at 99th percentile (~128 units) to remove bulk order distortion
- Exported `interactions.csv` and `metadata.csv`

---

## Method

### Association Rules (FP-Growth)
Basket matrix: `Invoice × StockCode`, binarised (bought = 1 / not bought = 0).

| Parameter | Value | Reasoning |
|---|---|---|
| min_support | 0.005 | ~245 co-occurrences minimum — filters noise without killing coverage |
| min_confidence | 0.50 | 50% conditional probability — rules must be meaningfully predictive |
| min_lift | 3.0 | At least 3× stronger than chance — removes popularity-inflated rules |

### TF-IDF Content-Based
Each product `Description` → TF-IDF vector. Cosine similarity computed across all 4,070 products.  
For each user, top-k similar products to each item in their history are surfaced, excluding already-purchased items. Maximum cosine score across all seed items is used as `content_score`.

### Hybrid Formula
```
hybrid_score = α × AR_score + (1 − α) × content_score
```
- `AR_score`: highest confidence among rules whose antecedent ⊆ user history
- `content_score`: max TF-IDF cosine similarity between candidate and user history

---

## Results

Evaluated on **1,000 customers** (min. 5 interactions), 80/20 train-test split per customer.

| α | Mode | Precision@5 | Recall@5 | F1@5 |
|---|---|---|---|---|
| 0.0 | Pure Content-Based | 0.0283 | 0.0248 | 0.0264 |
| 0.5 | Equal Hybrid | 0.0561 | 0.0331 | 0.0416 |
| 0.6 | AR-leaning | 0.0602 | 0.0353 | 0.0445 |
| 0.7 | AR-leaning | 0.0601 | 0.0348 | 0.0440 |
| 0.8 | Strong AR Bias | 0.0617 | 0.0352 | 0.0448 |
| **0.9** | **★ Best Hybrid** | **0.0637** | **0.0355** | **0.0453** |
| 1.0 | Pure AR | 0.0542 | 0.0242 | 0.0335 |

**Hybrid at α=0.9 outperforms:**
- Pure AR by **+35.2% F1**
- Pure Content-Based by **+71.6% F1**

---

## Setup

```bash
git clone https://github.com/hadyelfadaly/E-Commerce-Hybrid-Recommender.git
cd E-Commerce-Hybrid-Recommender/Code
pip install -r requirements.txt
```

Place the raw Online Retail II Excel file in the `Data/` directory, then run the preprocessing notebook before `script.py`.

```bash
python script.py
```

The script prompts for `Customer ID`, `alpha`, and `k` interactively.

---

## Team Members

- Hady El Fadaly - [Github Profile](https://github.com/hadyelfadaly)
- Yassin Mohy Eldin - [Github Profile](https://github.com/Yassin-Mohy)
- Ibrahim Wael El Noty - [Github Profile](https://github.com/ibrahimelnouty)
