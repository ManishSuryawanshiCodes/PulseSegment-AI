# PulseSegment AI — Customer Segmentation & Growth Intelligence

[![Next.js](https://img.shields.io/badge/Next.js-15.0-black?style=flat-square&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)

> **A Full-Stack, Dataset-Agnostic Data Mining & AI Decision-Support Platform Benchmarking 4 Unsupervised Clustering Algorithms Across 4 Feature Engineering & Dimensionality Reduction Pipelines ($4 \times 4 = 16$ Experiments).**

---

## Academic Metadata
- **Project Title:** AI-Powered Customer Segmentation and Growth Intelligence: A Comparative Analysis of Clustering Algorithms with Feature Selection and Dimensionality Reduction Techniques
- **Author:** Manish Pradeep Suryawanshi (PRN: `3542511013`)
- **Program:** Master of Computer Applications (MCA II)
- **Institution:** School of Computer Science and Applications, DES Pune University (DES PU), Pune, India
- **Course Guide:** Prof. Kakasaheb Nikam

---

## Overview & Business Problem
Modern retail banks, e-commerce brands, and SaaS enterprises store millions of high-dimensional behavioral logs. However, feeding raw, multi-collinear features directly into unsupervised clustering algorithms causes severe Euclidean distance distortion ("Curse of Dimensionality"), slow convergence, and overlapping customer segments.

**PulseSegment AI** solves this through a dataset-agnostic, two-stage architecture:
1. **Empirical Data Mining Benchmark Engine:** Executes a rigorous **16-experiment matrix** comparing **Original Numerical Features (DS1)**, **Correlation Filter (DS2, $|r| > 0.85$)**, **Mutual Information SelectKBest (DS3)**, and **Orthogonal PCA 95% Variance (DS4)** across **K-Means**, **Agglomerative Hierarchical**, **DBSCAN**, and **Gaussian Mixture Models (GMM)**.
2. **AI Growth Intelligence Layer:** Dynamically translates the optimal mathematical cluster centroids into actionable domain personas with automated **Next-Best-Action (NBA)** retention and growth campaigns.

> **Zero Hard-Coded Metrics:** No metric, record count, centroid, or persona percentage is hard-coded anywhere in the codebase. All metrics, cluster assignments, silhouette scores, and persona distributions are computed dynamically at runtime from the active dataset.

---

## Reference Results from the Original Study (Credit Card Baseline)

The table below documents the **historical reference values from the benchmark research report** on the UCI/Kaggle Credit Card dataset (CC GENERAL.csv, $N=8,950$ records, 18 features). In the live application, all figures are computed on demand:

| Dataset Representation | Dimensions | K-Means (Sil / DBI / Speed) | Hierarchical (Sil / DBI / Speed) | DBSCAN (Sil / DBI / Speed) | GMM (Sil / DBI / Speed) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Original Scaled (DS1)** | `8950 × 17` | `0.284` / `1.42` / `420ms` | `0.251` / `1.58` / `2850ms` | `0.192` / `1.89` / `1120ms` | `0.238` / `1.64` / `680ms` |
| **2. Correlation Filter (DS2)** | `8950 × 12` | `0.342` / `1.24` / `310ms` | `0.310` / `1.35` / `1940ms` | `0.245` / `1.62` / `780ms` | `0.305` / `1.39` / `490ms` |
| **3. Mutual Information (DS3)** | `8950 × 8` | `0.389` / `1.08` / `215ms` | `0.364` / `1.15` / `1320ms` | `0.312` / `1.34` / `510ms` | `0.358` / `1.18` / `340ms` |
| **4. PCA 95% Variance (DS4)** | `8950 × 6` | **`0.456` / `0.89` / `145ms`** | `0.418` / `0.98` / `980ms` | `0.375` / `1.12` / `340ms` | `0.432` / `0.94` / `230ms` |

---

## Run It on Your Own Data (Quick-Start Flow)

PulseSegment AI is 100% dataset-agnostic. Follow these 4 steps to segment any customer base:

### 1. Upload & Schema Auto-Detection (`Step 1: Data`)
- Drag and drop your customer CSV or Excel file (`.xlsx`) onto the dropzone (up to 50MB, $\ge 100$ rows, $\ge 3$ numeric features), or click **"Try the sample dataset"**.
- The engine automatically isolates identifier columns (`CUST_ID`, `user_id`), validates missing cells, and profiles data distribution statistics.
- Select your business domain template: **Banking / Credit Card**, **E-Commerce (RFM)**, **SaaS / Subscription**, or **Generic Business**.

### 2. Clean & Preprocess (`Step 2: Prepare`)
- Missing values are imputed with **KNN Imputation ($k=5$)**.
- Outliers are safely bounded with **Tukey IQR Winsorization ($1.5\times$ IQR capping)**.
- Features are scaled to zero mean and unit variance via **StandardScaler**.
- Inspect the interactive lower-triangle correlation heatmap with live threshold slider, and review pruned redundancies.

### 3. Run Benchmark (`Step 3: Benchmark`)
- Click **"Run 16-experiment benchmark"**.
- Watch the live execution progress panel test all 16 combinations (K-Means, Hierarchical, DBSCAN, and GMM across DS1–DS4).
- Review the **"Why This Model Won"** summary and examine the 16-row benchmark matrix with Simple (1–5 Star) or Expert (Silhouette, DBI, CHI, Speed) modes.

### 4. Explore & Act (`Step 4: Explore` & `Step 5: Act`)
- Inspect 2D PCA cluster projections ($PC1 \times PC2$) and K-sweep elbow curves ($k=2..10$).
- Review data-driven customer personas with diverging Z-score trait charts.
- Open the **Campaign Drawer** to plan personalized marketing initiatives and forecast incremental revenue with the **Interactive ROI Simulator**.
- **Export & Deploy:** Click **"Export Segments CSV"**, **"Download Model (.joblib)"**, or use **"Score One Customer"** to classify new prospects in real-time.

---

## Local Development & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ and pnpm

### Backend (FastAPI + Scikit-Learn)
```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation is available at `http://localhost:8000/docs`.

### Frontend (Next.js 15 App Router + Tailwind CSS)
```bash
cd frontend
pnpm dev
```
Interactive dashboard is available at `http://localhost:3000`.

### Running Automated Test Suite
```bash
cd backend
python -m pytest tests/test_pipeline.py -v
```
