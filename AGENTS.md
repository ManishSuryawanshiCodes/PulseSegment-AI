# AGENTS.md - Standing Directives for Google Antigravity Agents

## Project Overview
We are building "PulseSegment AI" — an enterprise-grade, dataset-agnostic Customer Segmentation and Growth Intelligence platform based on a 9-Phase Data Mining research rubric. The platform supports arbitrary uploaded customer tabular datasets (CSV / XLSX) as well as the standard benchmark Credit Card dataset (UCI/Kaggle CC GENERAL.csv, N=8,950 records, 18 features).

## CRITICAL DIRECTIVE: ZERO HARD-CODED METRICS OR RESULTS
- **Reference Values vs. Dynamic Runtime Computation:**
  - Any values cited in research documentation (e.g. Silhouette 0.456, DBI 0.89, CHI 3640.5, runtime 145ms, persona sizes 38.4%, 26.1%, 14.2%, 21.3%) are strictly **historical reference values from the benchmark report**, NEVER constants.
  - **NO metric, record count, missing count, percentage, centroid, ROI figure, persona size, or cluster label may be hard-coded anywhere in backend or frontend.**
  - Every calculation, metric, centroid, outlier count, variance ratio, and persona assignment MUST be computed at runtime from the active dataset.

## Design System: "Render / Linear Minimalist Dark & Light UI"
- **Color Palette:** Deep obsidian/zinc dark mode default (`#09090B` background, `#121215` elevated card surface, `#27272A` 1px crisp borders) with an optional clean light mode toggle (`#FAFAFA` background, `#FFFFFF` cards, `#E4E4E7` borders).
- **Accents:** Electric Indigo (`#6366F1`) for primary interactive states, Emerald (`#10B981`) for best clustering metrics, Amber (`#F59E0B`) for warnings/outliers, Rose (`#F43F5E`) for worst metrics.
- **Typography:** `Inter` or `Geist Sans` for UI text; `JetBrains Mono` or `Geist Mono` (`tabular-nums`) for all clustering metrics, execution times (ms), and cluster sizes.
- **Component Rules:**
  - NO bloated gradients or heavy drop shadows.
  - Use 1px borders (`border-zinc-800/80`), subtle top-edge inner highlights (`shadow-[inset_0_1px_0_0_rgba(255,255,255,0.05)]`), and frosted glass sticky headers (`backdrop-blur-md bg-zinc-950/75`).
  - Use pill-shaped live status indicators (e.g., green pulsing dot `Engine Ready • N Records Loaded`).
  - Visible "Synthetic Demo Data" badge when running on generated synthetic fallback.

## 9-Phase Data Mining Pipeline (Dataset-Agnostic)
1. **Schema & Role Auto-Detection:** Automatically detect ID columns, numerical features, categoricals, and constant columns with 10-row preview.
2. **Preprocessing Pipeline:**
   - KNN Imputation (k=5 default, configurable) for all numerical features with missing values.
   - Identifier exclusion (drop user-configured ID column).
   - Tukey IQR Winsorization (configurable multiplier, default 1.5x IQR) with before & after outlier counts computed dynamically.
   - `StandardScaler` (Z-score normalization: zero mean, unit variance).
3. **4 Feature Engineering Spaces:**
   - `DS1_Original`: All included scaled numerical features.
   - `DS2_Correlation`: Filter method removing collinear features with $|r| > \text{threshold}$ (default 0.85). Retains and prunes whatever the active data dictates; prunes only with a verified correlated partner.
   - `DS3_MutualInfo`: `SelectKBest(score_func=mutual_info_classif, k=min(8, n_features-1))` using KMeans pseudo-labels on DS1.
   - `DS4_PCA95`: `PCA(n_components=pca_variance, default 0.95)` with dynamic component count and full explained variance array.
4. **4 Clustering Algorithms (16 Experiments):**
   - `KMeans` ($k$ from config, default 4, k-means++)
   - `AgglomerativeClustering` ($n\_clusters=k$, linkage='ward', sampled if $N > 10,000$)
   - `DBSCAN` (auto-estimated $\epsilon$ from k-distance curve knee, min_samples from config default 15)
   - `GaussianMixture` ($n\_components=k$, covariance_type='full')
5. **Evaluation Metrics Logged per Run:** Silhouette Score (sampled if $N > 8,000$), Davies-Bouldin Index (DBI), Calinski-Harabasz Score (CHI), Execution Time (ms), Cluster Count, and Noise %.
6. **Winning Model Selection:** Composite score formula = mean of min-max normalized Silhouette, inverse DBI, and CHI.
7. **Personas & NBA (Data-Driven):**
   - Inverse-transformed cluster centroids in original units.
   - Z-score relative to global mean with top 3 positive and top 3 negative distinguishing attributes.
   - Domain templates (`credit_card`, `ecommerce_rfm`, `saas`, `generic`).
   - Dynamic ROI calculated from segment counts and user assumptions.
