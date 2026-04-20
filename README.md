# 🔍 Financial Fraud Detection at Scale
### Big Data Analytics Project — PySpark · Hadoop · Hive · LSTM

> Detecting fraud in 6.3 million financial transactions using a distributed ML pipeline and deep learning, achieving **ROC-AUC of 0.9999** with a Bidirectional LSTM.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Model Results](#model-results)
- [Key Findings](#key-findings)
- [Setup & Installation](#setup--installation)
- [Running the Notebook](#running-the-notebook)
- [Feature Engineering](#feature-engineering)
- [Model Details](#model-details)

---

## Overview

This project builds an end-to-end fraud detection system capable of processing millions of financial transactions in a distributed environment. It combines:

- A **distributed ML pipeline** (PySpark on a 3-node Hadoop cluster) training Random Forest and Gradient-Boosted Tree classifiers
- A **deep learning pipeline** (Bidirectional LSTM on GPU) that learns temporal patterns across transaction sequences
- **Apache Hive** for SQL-based exploratory analysis at scale
- **HDFS** for fault-tolerant distributed data storage

The project uses the [PaySim synthetic dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) — a simulation of mobile money transactions based on real financial logs — containing **6,362,620 transactions** with a fraud rate of only **0.13%**, making class imbalance a core challenge.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                             │
│             PaySim CSV (6.3M rows, ~470MB)                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HDFS / Hadoop Cluster                         │
│   master:9000  ──  3 nodes (master, worker1, worker2)           │
│   Raw CSV → Parquet (columnar, compressed, splittable)          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
┌─────────────────────┐  ┌─────────────────────────────────────────┐
│   Apache Hive DB    │  │     PySpark ML Pipeline                  │
│  fraud_detection_db │  │  StringIndexer → OneHotEncoder           │
│  SQL-based EDA      │  │  → VectorAssembler → StandardScaler      │
└─────────────────────┘  └────────────┬────────────────────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
              ┌─────────────────────┐   ┌────────────────────────┐
              │   Random Forest     │   │  Gradient-Boosted Trees │
              │   50 trees, d=10    │   │  50 iters, d=8          │
              │   ROC-AUC: 0.9979   │   │  ROC-AUC: 0.9965        │
              └─────────────────────┘   └────────────────────────┘

              ┌──────────────────────────────────────────────────┐
              │           Deep Learning Branch (GPU)              │
              │   Sliding Window Sequences (len=10, step=2)       │
              │   3.3M sequences → Bidirectional LSTM             │
              │   ROC-AUC: 0.9999  |  PR-AUC: 0.9995             │
              └──────────────────────────────────────────────────┘
```

---

## Dataset

**PaySim** — Simulated mobile money transactions

| Attribute | Value |
|-----------|-------|
| Total records | 6,362,620 |
| Fraud cases | 8,213 (0.13%) |
| Normal cases | 6,354,407 (99.87%) |
| Imbalance ratio | 773.7 : 1 |
| Time span | 744 hours (31 days) |
| Transaction types | CASH_OUT, PAYMENT, CASH_IN, TRANSFER, DEBIT |

**Columns:**

| Column | Description |
|--------|-------------|
| `step` | Hour of simulation (1–744) |
| `type` | Transaction type |
| `amount` | Transaction amount |
| `nameOrig` | Sender account ID |
| `oldbalanceOrg` / `newbalanceOrig` | Sender balance before/after |
| `nameDest` | Recipient account ID |
| `oldbalanceDest` / `newbalanceDest` | Recipient balance before/after |
| `isFraud` | Target label (0 = normal, 1 = fraud) |
| `isFlaggedFraud` | System flag (not used in modeling) |

> **Key insight:** Fraud occurs exclusively in `CASH_OUT` (50.1%) and `TRANSFER` (49.9%) transactions. `CASH_IN`, `PAYMENT`, and `DEBIT` have zero fraud cases.

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Distributed compute | Apache Spark 3.5.1 (PySpark) |
| Storage | HDFS (Hadoop), Parquet |
| SQL analytics | Apache Hive |
| Cluster manager | Spark Standalone (`spark://master:7077`) |
| ML (distributed) | PySpark MLlib (Random Forest, GBT) |
| ML (deep learning) | TensorFlow / Keras (Bidirectional LSTM) |
| Data processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Evaluation | scikit-learn |
| Environment | Jupyter Notebook |
| GPU (LSTM training) | 2× Tesla T4 (Kaggle) |

---

## Project Structure

```
BDA_Project/
├── data/
│   └── PS_20174392719_1491204439457_log.csv   # Raw PaySim dataset
├── models/
│   ├── random_forest_model/                    # Saved RF model (PySpark)
│   ├── gbt_model/                              # Saved GBT model (PySpark)
│   └── preprocessing_pipeline/                 # Saved preprocessing pipeline
├── output/
│   ├── plots/                                  # EDA & evaluation charts
│   └── metrics/
│       └── model_summary.html                  # Model comparison table
├── results/
│   └── deep_learning/
│       ├── models/
│       │   ├── lstm_fraud_model.keras          # Best LSTM weights
│       │   └── lstm_step_scaler.pkl            # Feature scaler
│       ├── plots/                              # LSTM training plots
│       └── metrics/
│           └── model_comparison_with_lstm.csv  # Final comparison
├── fraud_detection.ipynb                       # Main notebook (Cells 1–45)
└── README.md
```

---

## Pipeline Walkthrough

### Section 1 — Initialization (Cells 1–3)
- Library imports (PySpark, MLlib, sklearn, TensorFlow)
- Logging configuration
- SparkSession creation with Hive support, HDFS configuration, and multi-node executor validation

### Section 2 — Data Ingestion (Cells 4–8)
- Upload raw CSV to HDFS; convert to Parquet (columnar format, ~5× faster reads)
- Schema validation, null check (zero nulls found), and descriptive statistics

### Section 3 — Exploratory Data Analysis (Cells 9–15)
- Hive SQL queries for class imbalance, fraud-by-type, and fraud-by-amount analysis
- Visualizations: class distribution bar chart, fraud-by-type dual-axis chart, log-amount density plot

### Section 4 — Feature Engineering (Cells 16–18)
- Drop irrelevant columns (`nameOrig`, `nameDest`, `isFlaggedFraud`)
- Derive `hour_of_day` from `step` (`(step - 1) % 24`)
- Fraud-by-hour analysis (fraud is uniformly distributed across hours — no strong peak)

### Section 5 — Preprocessing (Cells 19–23)
- `StringIndexer` → `OneHotEncoder` for `type` column
- `VectorAssembler` assembles 12 features; `StandardScaler` normalizes
- Full pipeline fit on 6.3M rows, checkpointed to HDFS
- 80/20 train/test split; **oversampling** of fraud class in training set (776× ratio → ~50/50 balanced training data of 10.1M records)

### Section 6 — Model Training (Cells 24–25)
- **Random Forest:** 50 trees, max depth 10, subsampling rate 0.8, `sqrt` feature subset — trained in **19.2 minutes**
- **GBT:** 50 iterations, max depth 8, learning rate 0.1, subsampling 0.8 — trained in **110.7 minutes**

### Section 7 — Evaluation (Cells 27–31)
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
- ROC and Precision-Recall curves plotted for both models

### Section 8 — Advanced Analysis (Cells 32–34)
- Repeat fraudster analysis (SQL) — no repeat offenders found in dataset
- Fraud rate by amount range (highest rate: >$1M transactions at 2.07%)
- Hourly × transaction-type heatmaps for total and fraud transactions

### Section 10 — LSTM Deep Learning (Cells 38–45)
- Global sliding-window sequence generation over time-sorted data (3.18M windows)
- 8× oversampling of fraud windows with Gaussian noise augmentation → 3.33M sequences
- Per-step `StandardScaler`, stratified 70/15/15 train/val/test split
- Bidirectional LSTM (2 layers: 128→64 units), BatchNorm, Dropout, L2 regularization
- Training: 55 epochs (early stopping at patience=15), `ReduceLROnPlateau`, `val_auc` monitoring
- Optimal threshold tuned via F-beta (β=2) search on validation set → **threshold = 0.97**

---

## Model Results

### PySpark Models (Distributed, 3-Node Cluster)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Train Time |
|-------|----------|-----------|--------|----------|---------|--------|------------|
| Random Forest | 0.9830 | 0.9988 | 0.9830 | 0.9903 | **0.9979** | 0.9032 | 19.2 min |
| GBT | 0.9922 | 0.9989 | 0.9922 | 0.9951 | 0.9965 | 0.8955 | 110.7 min |

**Random Forest Confusion Matrix (test set: 1,272,827 records):**
```
              Predicted Normal   Predicted Fraud
Actual Normal    1,249,482           21,679
Actual Fraud            19            1,647
```

**GBT Confusion Matrix:**
```
              Predicted Normal   Predicted Fraud
Actual Normal    1,261,311            9,850
Actual Fraud            15            1,651
```

### Deep Learning — Bidirectional LSTM (GPU)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Train Time |
|-------|----------|-----------|--------|----------|---------|--------|------------|
| LSTM | **0.9996** | **0.9921** | **0.9993** | **0.9957** | **0.9999** | **0.9995** | 193.1 min |

**LSTM Confusion Matrix (test set: 500,244 sequences, threshold=0.97):**
```
              Predicted Normal   Predicted Fraud
Actual Normal       474,110              206
Actual Fraud             19           25,909
```

> The LSTM detects **99.93% of all fraud** while maintaining **99.21% precision** — only 206 false alarms out of 474,316 normal transactions.

---

## Key Findings

1. **Fraud type concentration:** 100% of fraud occurs in `CASH_OUT` and `TRANSFER` transactions. Filtering to these two types alone eliminates ~66% of data with zero fraud loss.

2. **Amount is the strongest signal:** Both tree models rank `oldbalanceOrg` and `newbalanceOrig` as the top-2 features, followed by `amount`. Balance patterns (e.g., account drained to zero) are highly discriminative.

3. **No temporal peak:** Fraud is distributed nearly uniformly across all 24 hours of the day — no "prime time" for fraudsters in this dataset.

4. **High-value transactions are highest risk:** Transactions over $1M have a fraud rate of **2.07%**, vs. 0.04% for transactions under $1K.

5. **No repeat offenders:** The analysis found zero accounts that committed more than one fraudulent transaction, suggesting sophisticated one-time fraud rather than repeat patterns.

6. **LSTM temporal advantage:** By modeling sequences of 10 consecutive transactions, the LSTM captures behavioral context that single-row classifiers miss — yielding a PR-AUC of **0.9995** vs. ~0.90 for the tree models.

7. **Class imbalance is critical:** With a 773:1 ratio, simple accuracy is misleading. PR-AUC is the most informative metric — the gap between GBT's PR-AUC (0.90) and LSTM's (0.9995) is substantial.

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- Java 8 or 11 (required for Spark/Hadoop)
- Apache Hadoop 3.x configured with HDFS
- Apache Spark 3.5.x
- Apache Hive 3.x (with metastore configured)

### Python Dependencies

```bash
pip install pyspark findspark pandas numpy matplotlib seaborn scikit-learn tensorflow joblib
```

### Hadoop Cluster Configuration

The notebook expects a Spark standalone cluster. Set environment variables before running:

```bash
export SPARK_MASTER="spark://master:7077"
export HDFS_BASE_PATH="/user/<your_username>/fraud_detection"
export SPARK_EXECUTOR_INSTANCES=2
export SPARK_EXECUTOR_CORES=2
export SPARK_EXECUTOR_MEMORY=4g
export SPARK_DRIVER_MEMORY=4g
```

For local testing only (bypasses cluster validation — edit Cell 3 to allow local mode):
```bash
export SPARK_MASTER="local[*]"
```

### Dataset

Download the PaySim dataset from Kaggle and place it at:
```
data/PS_20174392719_1491204439457_log.csv
```

[Download from Kaggle →](https://www.kaggle.com/datasets/ealaxi/paysim1)

---

## Running the Notebook

### Distributed Pipeline (Cells 1–37)

Run cells sequentially. The pipeline is **idempotent** — if HDFS Parquet files or trained models already exist, they are reused automatically.

**Expected runtimes on a 3-node cluster (2 cores × 4GB each):**
| Stage | Time |
|-------|------|
| CSV → HDFS Parquet | ~2 min |
| Preprocessing pipeline | ~25 min |
| Random Forest training | ~20 min |
| GBT training | ~2 hours |

### LSTM Pipeline (Cells 38–45)

The LSTM section is self-contained and reads directly from the CSV. It was developed and trained on Kaggle with 2× Tesla T4 GPUs.

**Expected runtimes:**
| Stage | Time |
|-------|------|
| Sequence generation (3.3M windows) | ~3 min |
| LSTM training (55 epochs, GPU) | ~3.2 hours |
| Evaluation + plotting | ~2 min |

To resume training from checkpoint, set `RESUME_LSTM = True` in Cell 38.

---

## Feature Engineering

| Feature | Source | Description |
|---------|--------|-------------|
| `type_vec` | `type` | One-hot encoded transaction type (5 categories) |
| `step` | raw | Hour of simulation |
| `amount` | raw | Transaction amount |
| `oldbalanceOrg` | raw | Sender balance before |
| `newbalanceOrig` | raw | Sender balance after |
| `oldbalanceDest` | raw | Recipient balance before |
| `newbalanceDest` | raw | Recipient balance after |
| `hour_of_day` | derived | `(step - 1) % 24` |
| `balance_change_orig` | derived | `newbalanceOrig - oldbalanceOrg` |
| `balance_change_dest` | derived | `newbalanceDest - oldbalanceDest` |
| `amount_to_balance_ratio` | derived | `amount / oldbalanceOrg` (0 if no balance) |

---

## Model Details

### Random Forest (PySpark MLlib)
```
numTrees=50, maxDepth=10, maxBins=32
subsamplingRate=0.8, featureSubsetStrategy="sqrt"
cacheNodeIds=True, checkpointInterval=10, seed=42
```

### Gradient-Boosted Trees (PySpark MLlib)
```
maxIter=50, maxDepth=8, stepSize=0.1
subsamplingRate=0.8, seed=42
```

### Bidirectional LSTM (TensorFlow/Keras)
```
Architecture:
  BiLSTM(128, return_sequences=True) → BatchNorm → Dropout(0.3)
  BiLSTM(64, return_sequences=False) → BatchNorm → Dropout(0.3)
  Dense(128, relu, L2=1e-4) → BatchNorm → Dropout(0.4)
  Dense(64, relu, L2=1e-4) → Dropout(0.3)
  Dense(32, relu) → Dropout(0.2)
  Dense(1, sigmoid)

Total parameters: 335,617 (1.28 MB)

Training config:
  Optimizer: Adam(lr=5e-4)
  Loss: BinaryCrossentropy(label_smoothing=0.01)
  Class weights: {0: 0.527, 1: 9.647}
  Early stopping: patience=15 on val_auc
  LR schedule: ReduceLROnPlateau(factor=0.5, patience=6)
  Best epoch: 40 / 55  |  Best val_auc: 0.999945
```

---

## 📊 Output Files

After running the full pipeline, the following artifacts are generated:

| File | Description |
|------|-------------|
| `output/plots/01_class_distribution.png` | Fraud vs. normal bar chart |
| `output/plots/02_fraud_by_type.png` | Fraud count and rate by transaction type |
| `output/plots/03_amount_distribution_log.png` | Log-amount density by class |
| `output/plots/05_fraud_by_hour_of_day.png` | Fraud count by hour |
| `output/plots/06_cm_Random_Forest.png` | RF confusion matrix |
| `output/plots/06_cm_GBT.png` | GBT confusion matrix |
| `output/plots/07_roc_pr_curves.png` | ROC & PR curves (RF vs GBT) |
| `output/plots/08_heatmap_total.png` | Hourly × type transaction heatmap |
| `output/plots/09_heatmap_fraud.png` | Hourly × type fraud heatmap |
| `output/metrics/model_summary.html` | Styled model comparison table |
| `results/deep_learning/plots/lstm_loss_accuracy.png` | LSTM training curves |
| `results/deep_learning/plots/lstm_confusion_matrix.png` | LSTM confusion matrix |
| `results/deep_learning/plots/lstm_roc_pr_curves.png` | LSTM ROC & PR curves |
| `results/deep_learning/metrics/model_comparison_with_lstm.csv` | Final metrics CSV |
