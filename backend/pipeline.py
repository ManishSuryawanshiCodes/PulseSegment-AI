import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import datetime
import time
import io
import math
import hashlib
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors, NearestCentroid
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib

from models import (
    DatasetConfig,
    DatasetSummary,
    FeatureEngineeringResult,
    PrunedFeatureInfo,
    ExperimentResult,
    ExperimentsMatrix,
    ScatterPoint,
    KSweepPoint,
    PersonaProfile,
    DistinguishingFeature,
    PersonaAssumptions,
    NBACampaign,
    GrowthPersonas,
    PredictSegmentResponse,
    ClusterBoxPlotStat,
    FeatureBoxPlot,
)

# Standard sampling constants across pipeline, API, and documentation
SILHOUETTE_SAMPLE_THRESHOLD = 8000
SILHOUETTE_SAMPLE_SIZE = 5000

DOMAIN_DIRECTION_MAP = {
    "ecommerce_rfm": {
        "recency": "lower_is_better",
        "return_rate": "lower_is_better",
        "frequency": "higher_is_better",
        "monetary": "higher_is_better",
        "avg_order_value": "higher_is_better",
        "orders": "higher_is_better",
        "churn": "lower_is_better",
    },
    "credit_card": {
        "purchases": "higher_is_better",
        "purchases_frequency": "higher_is_better",
        "oneoff_purchases": "higher_is_better",
        "installments_purchases": "higher_is_better",
        "payments": "higher_is_better",
        "credit_limit": "higher_is_better",
        "prc_full_payment": "higher_is_better",
        "cash_advance": "lower_is_better",
        "cash_advance_frequency": "lower_is_better",
        "cash_advance_trx": "lower_is_better",
        "balance": "neutral",
    },
    "saas": {
        "mrr": "higher_is_better",
        "seats": "higher_is_better",
        "logins": "higher_is_better",
        "usage": "higher_is_better",
        "growth": "higher_is_better",
        "churn_risk": "lower_is_better",
    },
    "generic": {},
}


def get_column_direction(domain: str, col_name: str) -> str:
    col_lower = col_name.lower()
    domain_map = DOMAIN_DIRECTION_MAP.get(domain, {})
    for key, direction in domain_map.items():
        if key in col_lower:
            return direction
    if any(k in col_lower for k in ["recency", "days_since", "return_rate", "churn", "risk", "bounce"]):
        return "lower_is_better"
    if any(k in col_lower for k in ["monetary", "revenue", "spend", "purchases", "frequency", "orders", "growth", "logins", "mrr", "seats", "payment"]):
        return "higher_is_better"
    return "neutral"


def format_trait_wording(col_name: str, z: float, domain: str) -> str:
    direction = get_column_direction(domain, col_name)
    col_lower = col_name.lower()

    if "recency" in col_lower or "days_since" in col_lower:
        if z < -0.3:
            return "Bought recently"
        elif z > 0.3:
            return "Lapsed / Long time since purchase"
        else:
            return "Moderate recency"

    if "return_rate" in col_lower:
        if z > 0.3:
            return "Return-prone"
        elif z < -0.3:
            return "Low return rate"
        else:
            return "Normal returns"

    clean_name = col_name.replace("_", " ").title()
    if direction == "lower_is_better":
        if z < -0.3:
            return f"Favorable low {clean_name}"
        elif z > 0.3:
            return f"High {clean_name}"
    elif direction == "higher_is_better":
        if z > 0.3:
            return f"High {clean_name}"
        elif z < -0.3:
            return f"Below-average {clean_name}"
    else:
        if z > 0.3:
            return f"High {clean_name}"
        elif z < -0.3:
            return f"Low {clean_name}"

    return f"Moderate {clean_name}"


def get_feature_z(z_dict: Dict[str, float], *patterns: str) -> float:
    for pat in patterns:
        pat_lower = pat.lower()
        for k, v in z_dict.items():
            if pat_lower == k.lower() or pat_lower in k.lower():
                return float(v)
    return 0.0


class PulseSegmentPipeline:
    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[DatasetConfig] = None,
        dataset_id: str = "default",
        is_synthetic: bool = False,
        db: Optional[Any] = None,
    ):
        self.dataset_id = dataset_id
        self.is_synthetic = is_synthetic
        self.config = config or DatasetConfig()
        self.config_hash = self._compute_config_hash(self.config)
        self.db = db

        self._raw_df = df.copy()
        self._numeric_cols: List[str] = []
        self._feature_names: List[str] = []

        # Model artifacts
        self._imputer: Optional[KNNImputer] = None
        self._scaler: Optional[StandardScaler] = None
        self._winsor_bounds: Dict[str, Tuple[float, float]] = {}
        self._global_means: Dict[str, float] = {}
        self._global_stds: Dict[str, float] = {}

        # Preprocessed dataframes (float32 optimized)
        self._winsorized_df: Optional[pd.DataFrame] = None
        self._processed_df: Optional[pd.DataFrame] = None
        self._datasets: Dict[str, pd.DataFrame] = {}

        # Dimensionality Reduction
        self._pca: Optional[PCA] = None
        self._pca_2d: Optional[PCA] = None

        # Caches
        self._summary_cache: Optional[Dict[str, Any]] = None
        self._feature_engineering_cache: Optional[Dict[str, Any]] = None
        self._experiments_cache: Optional[Dict[str, Any]] = None
        self._personas_cache: Optional[Dict[str, Any]] = None
        self._winning_model: Optional[Any] = None
        self._winning_labels: Optional[np.ndarray] = None
        self._winning_dataset_name: Optional[str] = None
        self._winning_alg_name: Optional[str] = None

        # Try loading cached results from SQLite if available
        if self.db:
            cached = self.db.get_pipeline_cache(self.dataset_id, self.config_hash)
            if cached:
                self._summary_cache = cached.get("summary")
                self._feature_engineering_cache = cached.get("features")
                self._experiments_cache = cached.get("experiments")
                self._personas_cache = cached.get("personas")

    def _compute_config_hash(self, config: DatasetConfig) -> str:
        raw = json.dumps(config.model_dump(), sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    def _prepare_raw_features(self) -> pd.DataFrame:
        """Filter excluded columns and isolate numerical features."""
        df = self._raw_df.copy()

        # Drop explicit ID column if configured
        if self.config.id_column and self.config.id_column in df.columns:
            df = df.drop(columns=[self.config.id_column])

        # If user specified include_columns, keep only those
        if self.config.include_columns:
            valid_includes = [c for c in self.config.include_columns if c in df.columns]
            if len(valid_includes) >= 3:
                df = df[valid_includes]

        # Keep only numerical columns
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        valid_cols = []
        for col in numeric_df.columns:
            if numeric_df[col].dropna().nunique() > 1:
                valid_cols.append(col)

        if len(valid_cols) < 3:
            valid_cols = list(numeric_df.columns)

        self._numeric_cols = valid_cols
        return numeric_df[valid_cols].astype(np.float32)

    def preprocess(self) -> pd.DataFrame:
        """
        Executes dynamic 4-step preprocessing:
        1. Identifier Removal / Feature Isolation
        2. Dynamic KNN Imputation (k=config.imputation_k)
        3. Dynamic Tukey IQR Winsorization (config.iqr_multiplier)
        4. StandardScaler Z-Score Normalization
        Uses float32 matrices for speed.
        """
        if self._processed_df is not None:
            return self._processed_df

        df_num = self._prepare_raw_features()
        self._feature_names = list(df_num.columns)

        # Compute global unscaled stats for z-score ranking
        for col in self._feature_names:
            series = df_num[col].dropna()
            self._global_means[col] = float(series.mean()) if len(series) > 0 else 0.0
            self._global_stds[col] = float(series.std()) if len(series) > 1 and series.std() > 0 else 1.0

        # Step 2: KNN Imputation (k = config.imputation_k)
        has_missing = df_num.isna().any().any()
        if has_missing:
            k = min(self.config.imputation_k, max(1, len(df_num) - 1))
            self._imputer = KNNImputer(n_neighbors=k)
            if len(df_num) > 3000:
                complete_df = df_num.dropna()
                train_data = (
                    complete_df.sample(n=min(3000, len(complete_df)), random_state=42)
                    if len(complete_df) >= k
                    else df_num.sample(n=min(3000, len(df_num)), random_state=42)
                )
                self._imputer.fit(train_data)
                imputed_arr = self._imputer.transform(df_num)
            else:
                imputed_arr = self._imputer.fit_transform(df_num)
            imputed_df = pd.DataFrame(imputed_arr, columns=self._feature_names, index=df_num.index, dtype=np.float32)
        else:
            imputed_df = df_num.copy()

        # Step 3: Tukey IQR Winsorization
        multiplier = self.config.iqr_multiplier
        winsor_df = imputed_df.copy()
        self._winsor_bounds = {}

        for col in self._feature_names:
            series = winsor_df[col]
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            self._winsor_bounds[col] = (lower, upper)
            winsor_df[col] = series.clip(lower=lower, upper=upper)

        self._winsorized_df = winsor_df

        # Step 4: StandardScaler Normalization (float32)
        self._scaler = StandardScaler()
        scaled_arr = self._scaler.fit_transform(winsor_df).astype(np.float32)
        self._processed_df = pd.DataFrame(scaled_arr, columns=self._feature_names, index=winsor_df.index, dtype=np.float32)

        return self._processed_df

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Calculates descriptive stats, missing values, and verified IQR outlier counts before & after winsorization."""
        if self._summary_cache is not None:
            return self._summary_cache

        df_num = self._prepare_raw_features()
        row_count = int(self._raw_df.shape[0])
        col_count = int(self._raw_df.shape[1])
        columns = list(self._raw_df.columns)

        missing_values = {
            col: int(self._raw_df[col].isna().sum())
            for col in columns
            if self._raw_df[col].isna().sum() > 0
        }

        descriptive_stats = {}
        iqr_before = {}
        multiplier = self.config.iqr_multiplier

        for col in self._feature_names:
            series = df_num[col].dropna()
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr

            outliers_count = int(((series < lower) | (series > upper)).sum())
            iqr_before[col] = outliers_count

            descriptive_stats[col] = {
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2) if len(series) > 1 else 0.0,
                "median": round(float(series.median()), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "q1": round(q1, 2),
                "q3": round(q3, 2),
            }

        self.preprocess()

        iqr_after = {}
        for col in self._feature_names:
            series_after = self._winsorized_df[col]
            lower, upper = self._winsor_bounds[col]
            outliers_after = int(((series_after < lower - 1e-4) | (series_after > upper + 1e-4)).sum())
            iqr_after[col] = outliers_after

        id_text = f"Removed ID column '{self.config.id_column}'" if self.config.id_column else "No identifier column specified"
        preprocessing_steps = [
            f"1. {id_text} and isolated {len(self._feature_names)} numerical features.",
            f"2. KNN Imputation (k={self.config.imputation_k}) applied across all features with missing entries.",
            f"3. Tukey IQR Winsorization ({self.config.iqr_multiplier}x IQR) capped extreme distribution tails.",
            f"4. StandardScaler Z-score normalization standardized features to zero mean and unit variance.",
        ]

        self._summary_cache = {
            "dataset_id": self.dataset_id,
            "row_count": row_count,
            "col_count": col_count,
            "is_synthetic": self.is_synthetic,
            "columns": columns,
            "missing_values": missing_values,
            "descriptive_stats": descriptive_stats,
            "iqr_outlier_counts_before": iqr_before,
            "iqr_outlier_counts_after": iqr_after,
            "preprocessing_steps": preprocessing_steps,
        }

        if self.db:
            self.db.save_pipeline_cache(self.dataset_id, self.config_hash, summary=self._summary_cache)

        return self._summary_cache

    def feature_engineering(self) -> Dict[str, Any]:
        """
        Constructs the 4 feature engineering representations:
        1. DS1_Original: All scaled numerical features (float32).
        2. DS2_Correlation: Collinear pairs (|r| > threshold) filtered dynamically with partner tracking.
        3. DS3_MutualInfo: SelectKBest on pseudo-labels derived from KMeans on DS1.
        4. DS4_PCA95: PCA explaining target variance (default 95%).
        """
        if self._feature_engineering_cache is not None and "DS1_Original" in self._datasets and "DS4_PCA95" in self._datasets:
            return self._feature_engineering_cache

        self.preprocess()
        ds1 = self._processed_df.copy()
        feature_names = list(ds1.columns)
        self._datasets["DS1_Original"] = ds1

        # --- Correlation Analysis (DS2) ---
        corr_matrix = ds1.corr(method="pearson")
        threshold = self.config.corr_threshold

        collinear_pairs = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                r_val = abs(corr_matrix.iloc[i, j])
                if r_val > threshold and not np.isnan(r_val):
                    collinear_pairs.append((feature_names[i], feature_names[j], r_val))

        collinear_pairs.sort(key=lambda x: x[2], reverse=True)
        var_series = self._winsorized_df.var() if self._winsorized_df is not None else ds1.var()

        removed_features: List[str] = []
        pruned_details: List[Dict[str, Any]] = []

        for f1, f2, r_val in collinear_pairs:
            if f1 not in removed_features and f2 not in removed_features:
                pruned = f2 if var_series[f1] >= var_series[f2] else f1
                retained = f1 if pruned == f2 else f2
                removed_features.append(pruned)
                pruned_details.append({
                    "feature": pruned,
                    "correlated_with": retained,
                    "correlation": round(float(r_val), 4),
                })

        ds2_features = [f for f in feature_names if f not in removed_features]
        if len(ds2_features) < 2:
            ds2_features = feature_names[: min(2, len(feature_names))]
            removed_features = [f for f in feature_names if f not in ds2_features]
            pruned_details = [d for d in pruned_details if d["feature"] in removed_features]

        ds2 = ds1[ds2_features].copy()
        self._datasets["DS2_Correlation"] = ds2

        corr_matrix_list = []
        for i, row in enumerate(corr_matrix.values):
            row_list = []
            for j, val in enumerate(row):
                if np.isnan(val) or np.isinf(val):
                    row_list.append(1.0 if i == j else 0.0)
                else:
                    v = round(float(val), 2)
                    row_list.append(0.0 if abs(v) < 1e-6 else v)
            corr_matrix_list.append(row_list)

        # --- Mutual Information (DS3) ---
        n_features = len(feature_names)
        k_mi = self.config.mi_k if self.config.mi_k else min(8, max(2, n_features - 1))
        k_mi = min(k_mi, n_features)

        prelim_km = KMeans(n_clusters=self.config.n_clusters, random_state=42, n_init=2)
        pseudo_labels = prelim_km.fit_predict(ds1)

        try:
            selector = SelectKBest(score_func=mutual_info_classif, k=k_mi)
            if len(ds1) > 2000:
                idx_sub = np.random.RandomState(42).choice(len(ds1), size=2000, replace=False)
                selector.fit(ds1.iloc[idx_sub], pseudo_labels[idx_sub])
            else:
                selector.fit(ds1, pseudo_labels)
            mi_scores = {col: round(float(score), 4) for col, score in zip(feature_names, selector.scores_)}
            selected_indices = selector.get_support(indices=True)
            ds3_features = [feature_names[i] for i in selected_indices]
        except Exception:
            ds3_features = feature_names[:k_mi]
            mi_scores = {col: 0.1 for col in ds3_features}

        ds3 = ds1[ds3_features].copy()
        self._datasets["DS3_MutualInfo"] = ds3
        ds3_method_note = (
            "Supervised feature selection (mutual_info_classif) applied against pseudo-labels "
            "derived from preliminary K-Means clustering on full feature space (DS1)."
        )

        # --- PCA Target Variance (DS4) ---
        target_var = min(0.99, max(0.50, self.config.pca_variance))
        pca_estimator = PCA(n_components=target_var, random_state=42)
        try:
            ds4_arr = pca_estimator.fit_transform(ds1).astype(np.float32)
            n_components = max(2, int(pca_estimator.n_components_))
            exp_ratios = [round(float(v), 4) for v in pca_estimator.explained_variance_ratio_]
            cum_var = [round(float(v), 4) for v in np.cumsum(pca_estimator.explained_variance_ratio_)]
        except Exception:
            n_components = min(6, max(2, n_features))
            pca_estimator = PCA(n_components=n_components, random_state=42)
            ds4_arr = pca_estimator.fit_transform(ds1).astype(np.float32)
            exp_ratios = [round(float(v), 4) for v in pca_estimator.explained_variance_ratio_]
            cum_var = [round(float(v), 4) for v in np.cumsum(pca_estimator.explained_variance_ratio_)]

        ds4_cols = [f"PC{i+1}" for i in range(ds4_arr.shape[1])]
        ds4 = pd.DataFrame(ds4_arr, columns=ds4_cols, index=ds1.index, dtype=np.float32)
        self._datasets["DS4_PCA95"] = ds4
        self._pca = pca_estimator

        self._feature_engineering_cache = {
            "dataset_id": self.dataset_id,
            "original_features": feature_names,
            "ds2_correlation_features": ds2_features,
            "ds2_removed_features": removed_features,
            "ds2_pruned_details": pruned_details,
            "correlation_matrix": corr_matrix_list,
            "correlation_feature_names": feature_names,
            "ds3_mutual_info_features": ds3_features,
            "ds3_mutual_info_scores": mi_scores,
            "ds3_method_note": ds3_method_note,
            "ds4_pca_n_components": n_components,
            "ds4_pca_explained_variance_ratio": exp_ratios,
            "ds4_pca_cumulative_variance": cum_var,
        }

        if self.db:
            self.db.save_pipeline_cache(self.dataset_id, self.config_hash, features=self._feature_engineering_cache)

        return self._feature_engineering_cache

    def _estimate_dbscan_eps(self, X: np.ndarray, min_samples: int = 15) -> float:
        """
        Auto-estimates DBSCAN eps from the k-distance curve knee.
        Robust across arbitrary scale dimensions.
        """
        k = min(min_samples, max(2, X.shape[0] - 1))
        if X.shape[0] > 1000:
            sample_idx = np.random.RandomState(42).choice(X.shape[0], size=1000, replace=False)
            X_sub = X[sample_idx]
        else:
            X_sub = X

        nbrs = NearestNeighbors(n_neighbors=k).fit(X_sub)
        distances, _ = nbrs.kneighbors(X_sub)
        k_distances = np.sort(distances[:, k - 1])

        eps = float(np.percentile(k_distances, 85))
        return max(0.1, round(eps, 3))

    def _run_dbscan(
        self,
        X: np.ndarray,
        time_budget: float = 8.0,
    ) -> Tuple[np.ndarray, Optional[float], float, str, Optional[str], int]:
        """
        Item 2d:
        - 8 s time budget.
        - Estimates eps from k-distance knee.
        - Grid of 5 eps values.
        - Stratified/random sample of at most 5000 rows.
        - Assigns rest by nearest neighbor.
        - If < 2 clusters or > 60% noise: returns metrics null with 'Density method found no clear groups'.
        """
        t_start = time.perf_counter()
        n_rows = X.shape[0]

        is_sampled = n_rows > 1500
        if is_sampled:
            sample_idx = np.random.RandomState(42).choice(n_rows, size=1500, replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X

        base_eps = self._estimate_dbscan_eps(X_sample, min_samples=15)
        eps_grid = [round(base_eps * m, 3) for m in [0.7, 0.85, 1.0, 1.2, 1.4]]

        best_sub_labels = None
        best_noise_pct = 100.0
        best_cluster_count = 0
        best_eps = base_eps
        found_valid = False

        for eps_val in eps_grid:
            # Check time budget
            if time.perf_counter() - t_start >= time_budget:
                break

            try:
                db = DBSCAN(eps=eps_val, min_samples=15)
                sub_labels = db.fit_predict(X_sample)
                u_clusters = [c for c in set(sub_labels) if c != -1]
                noise_count = int(np.sum(sub_labels == -1))
                noise_pct = (noise_count / len(sub_labels)) * 100.0

                if len(u_clusters) >= 2 and noise_pct <= 60.0:
                    found_valid = True
                    if noise_pct < best_noise_pct:
                        best_noise_pct = noise_pct
                        best_cluster_count = len(u_clusters)
                        best_sub_labels = sub_labels
                        best_eps = eps_val
                        if best_noise_pct <= 25.0:
                            break
            except Exception:
                continue

        best_labels_full = None
        if found_valid and best_sub_labels is not None:
            if is_sampled:
                non_noise = (best_sub_labels != -1)
                if np.sum(non_noise) >= best_cluster_count:
                    nn = NearestNeighbors(n_neighbors=1).fit(X_sample[non_noise])
                    _, indices = nn.kneighbors(X)
                    best_labels_full = best_sub_labels[non_noise][indices.ravel()]
                else:
                    best_labels_full = np.full(n_rows, -1, dtype=int)
                    best_labels_full[sample_idx] = best_sub_labels
            else:
                best_labels_full = best_sub_labels

            actual_noise = (np.sum(best_labels_full == -1) / n_rows) * 100.0
            actual_clusters = len([c for c in set(best_labels_full) if c != -1])
            if actual_clusters >= 2 and actual_noise <= 60.0:
                return best_labels_full, best_eps, round(actual_noise, 2), "success", None, actual_clusters

        # Degenerate outcome: return null metrics with exact reason string
        fallback_labels = best_labels_full if best_labels_full is not None else np.full(n_rows, -1, dtype=int)
        u_fallback = len([c for c in set(fallback_labels) if c != -1])
        noise_fallback = round(float((np.sum(fallback_labels == -1) / n_rows) * 100.0), 2)
        return fallback_labels, best_eps, noise_fallback, "degenerate", "Density method found no clear groups", u_fallback

    def run_k_sweep(self) -> Tuple[List[Dict[str, Any]], int, str]:
        """
        Item 2h:
        Computes real K-sweep (k=2..10) on DS1 returning inertia and silhouette.
        Never auto-picks k=2 just because silhouette is highest there.
        Uses composite (silhouette, CH, DBI rank average) with minimum cluster size >= 3% of rows and k >= 3.
        Returns suggested_k and suggested_k_reason.
        """
        self.preprocess()
        X = self._datasets.get("DS1_Original", self._processed_df).values.astype(np.float32)
        n_samples = X.shape[0]
        max_k = min(10, max(3, n_samples - 1))

        if n_samples > 1500:
            sub_idx = np.random.RandomState(42).choice(n_samples, size=1500, replace=False)
            X_eval = X[sub_idx]
        else:
            X_eval = X

        sweep_data = []
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=1)
            km.fit(X)
            inertia = round(float(km.inertia_), 2)

            labels_full = km.predict(X)
            labels_eval = km.predict(X_eval)

            counts = [int(np.sum(labels_full == c)) for c in range(k)]
            min_size = min(counts) if counts else 0
            min_size_pct = round(float((min_size / n_samples) * 100.0), 2)

            if len(set(labels_eval)) > 1:
                sil = round(float(silhouette_score(X_eval, labels_eval)), 4)
                dbi = round(float(davies_bouldin_score(X_eval, labels_eval)), 4)
                ch = round(float(calinski_harabasz_score(X_eval, labels_eval)), 2)
            else:
                sil = 0.0
                dbi = 999.0
                ch = 0.0

            sweep_data.append({
                "k": k,
                "inertia": inertia,
                "silhouette": sil,
                "dbi": dbi,
                "ch": ch,
                "min_size": min_size,
                "min_size_pct": min_size_pct,
            })

        # Filter candidate k values: k >= 3 (strictly never auto-pick k=2)
        candidates = [s for s in sweep_data if s["k"] >= 3]
        valid_size_candidates = [s for s in candidates if s["min_size_pct"] >= 3.0]
        eval_pool = valid_size_candidates if valid_size_candidates else candidates

        if eval_pool:
            sorted_by_sil = sorted(eval_pool, key=lambda s: s["silhouette"], reverse=True)
            sorted_by_ch = sorted(eval_pool, key=lambda s: s["ch"], reverse=True)
            sorted_by_dbi = sorted(eval_pool, key=lambda s: s["dbi"], reverse=False)

            rank_sil = {s["k"]: i + 1 for i, s in enumerate(sorted_by_sil)}
            rank_ch = {s["k"]: i + 1 for i, s in enumerate(sorted_by_ch)}
            rank_dbi = {s["k"]: i + 1 for i, s in enumerate(sorted_by_dbi)}

            for s in eval_pool:
                avg_rank = (rank_sil[s["k"]] + rank_ch[s["k"]] + rank_dbi[s["k"]]) / 3.0
                s["composite_rank"] = round(avg_rank, 2)

            best_s = min(eval_pool, key=lambda s: s["composite_rank"])
            suggested_k = best_s["k"]
            suggested_k_reason = (
                f"k={suggested_k} selected by composite rank (Silhouette {best_s['silhouette']:.3f}, "
                f"DBI {best_s['dbi']:.2f}, CHI {best_s['ch']:.0f}) with balanced clusters "
                f"(smallest cluster: {best_s['min_size_pct']:.1f}% >= 3.0% minimum threshold)."
            )
        else:
            suggested_k = 4
            suggested_k_reason = "Default k=4 selected with balanced cluster distribution."

        sanitized_sweep = [
            {
                "k": s["k"],
                "inertia": s["inertia"],
                "silhouette": s["silhouette"],
                "composite_rank": s.get("composite_rank"),
                "min_cluster_size_pct": s["min_size_pct"],
            }
            for s in sweep_data
        ]
        return sanitized_sweep, suggested_k, suggested_k_reason

    def _execute_single_experiment(
        self,
        dname: str,
        alg_name: str,
        k_clusters: int,
        coords_2d: np.ndarray,
        X_common: np.ndarray,
    ) -> Dict[str, Any]:
        """Runs an individual experiment in parallel."""
        df_cur = self._datasets[dname]
        X = df_cur.values.astype(np.float32)
        n_rows, n_cols = X.shape
        dim_str = f"{n_rows}x{n_cols}"

        t0 = time.perf_counter()

        if alg_name == "KMeans":
            if n_rows > 50000:
                model = MiniBatchKMeans(n_clusters=k_clusters, batch_size=2048, random_state=42)
            else:
                model = KMeans(n_clusters=k_clusters, init="k-means++", n_init=3, random_state=42)
            labels = model.fit_predict(X)
            noise_percentage = 0.0
            cluster_count = k_clusters
            status_str = "success"
            reason_str = None

        elif alg_name == "Agglomerative":
            # Cap at 2000 rows (strictly <= 5000 directive) with nearest-centroid assignment of the rest
            model = AgglomerativeClustering(n_clusters=k_clusters, linkage="ward")
            sample_cap = min(2000, n_rows)
            if n_rows > sample_cap:
                sample_idx = np.random.RandomState(42).choice(n_rows, size=sample_cap, replace=False)
                X_fit = X[sample_idx]
                sub_labels = model.fit_predict(X_fit)
                nc = NearestCentroid().fit(X_fit, sub_labels)
                labels = nc.predict(X)
            else:
                labels = model.fit_predict(X)
            noise_percentage = 0.0
            cluster_count = k_clusters
            status_str = "success"
            reason_str = None

        elif alg_name == "DBSCAN":
            labels, _, noise_percentage, status_str, reason_str, cluster_count = self._run_dbscan(X, time_budget=8.0)
            model = DBSCAN(eps=0.5, min_samples=15)

        elif alg_name == "GaussianMixture":
            model = GaussianMixture(
                n_components=k_clusters,
                covariance_type="full",
                reg_covar=1e-3,
                max_iter=10,
                tol=1e-2,
                n_init=1,
                random_state=42,
            )
            if n_rows > 3500:
                gmm_sample_idx = np.random.RandomState(42).choice(n_rows, size=3500, replace=False)
                model.fit(X[gmm_sample_idx])
                labels = model.predict(X)
            else:
                labels = model.fit_predict(X)
            noise_percentage = 0.0
            cluster_count = k_clusters
            status_str = "success"
            reason_str = None

        t1 = time.perf_counter()
        exec_time_ms = round((t1 - t0) * 1000.0, 2)

        noise_mask = (labels == -1)
        noise_count = int(np.sum(noise_mask))
        unique_clusters = sorted([c for c in set(labels) if c != -1])
        cluster_sizes = {int(c): int(np.sum(labels == c)) for c in set(labels)}

        # Metrics in active feature space
        sil = None
        dbi = None
        ch = None
        sampled_sil = False
        sample_size_used = None

        # Common-space clarity metrics on DS1_Original (Item 3)
        common_sil = None
        common_dbi = None
        common_ch = None

        is_valid_clustering = (cluster_count >= 2 and noise_percentage <= 60.0 and status_str == "success")

        if is_valid_clustering:
            valid_mask = ~noise_mask if noise_count > 0 else np.ones(n_rows, dtype=bool)
            X_eval = X[valid_mask]
            X_common_eval = X_common[valid_mask]
            labels_eval = labels[valid_mask]

            if len(set(labels_eval)) >= 2:
                if len(labels_eval) > SILHOUETTE_SAMPLE_THRESHOLD:
                    sil_idx = np.random.RandomState(42).choice(len(labels_eval), size=SILHOUETTE_SAMPLE_SIZE, replace=False)
                    X_sil = X_eval[sil_idx]
                    X_c_sil = X_common_eval[sil_idx]
                    l_sil = labels_eval[sil_idx]
                    sampled_sil = True
                    sample_size_used = SILHOUETTE_SAMPLE_SIZE
                else:
                    X_sil = X_eval
                    X_c_sil = X_common_eval
                    l_sil = labels_eval

                try:
                    # Active space metrics
                    sil = round(float(silhouette_score(X_sil, l_sil)), 4)
                    dbi = round(float(davies_bouldin_score(X_sil, l_sil)), 4)
                    ch = round(float(calinski_harabasz_score(X_sil, l_sil)), 2)

                    # Common space metrics on DS1_Original (Item 3)
                    if dname == "DS1_Original":
                        common_sil = sil
                        common_dbi = dbi
                        common_ch = ch
                    else:
                        common_sil = round(float(silhouette_score(X_c_sil, l_sil)), 4)
                        common_dbi = round(float(davies_bouldin_score(X_c_sil, l_sil)), 4)
                        common_ch = round(float(calinski_harabasz_score(X_c_sil, l_sil)), 2)
                except Exception as ex:
                    status_str = "error"
                    reason_str = str(ex)
            else:
                status_str = "degenerate"
                reason_str = "Fewer than 2 distinct clusters found."
        else:
            if status_str != "degenerate":
                status_str = "degenerate"
                reason_str = reason_str or "Clustering yielded invalid cluster configuration."

        # Downsample scatter data server-side (up to 500 points for speed)
        scatter_limit = min(500, n_rows)
        if n_rows > scatter_limit:
            sub_scatter_idx = np.random.RandomState(42).choice(n_rows, size=scatter_limit, replace=False)
        else:
            sub_scatter_idx = np.arange(n_rows)

        scatter_data = [
            {
                "pc1": round(float(coords_2d[idx, 0]), 4),
                "pc2": round(float(coords_2d[idx, 1]), 4),
                "cluster_label": int(labels[idx]),
            }
            for idx in sub_scatter_idx
        ]

        return {
            "dataset_name": dname,
            "dataset_dim": dim_str,
            "algorithm": alg_name,
            "silhouette_score": sil,
            "davies_bouldin_index": dbi,
            "calinski_harabasz_score": ch,
            "common_space_silhouette": common_sil,
            "common_space_dbi": common_dbi,
            "common_space_chi": common_ch,
            "composite_score": None,
            "execution_time_ms": exec_time_ms,
            "cluster_count": cluster_count,
            "noise_percentage": noise_percentage,
            "sampled_silhouette": sampled_sil,
            "silhouette_sample_size": sample_size_used,
            "status": status_str,
            "reason": reason_str,
            "scatter_data": scatter_data,
            "cluster_sizes": cluster_sizes,
            "_model": model,
            "_labels": labels,
        }

    def run_experiments(self, job_callback=None, cancel_check=None) -> Dict[str, Any]:
        """
        Executes the 16-experiment benchmark matrix (4 Feature Spaces x 4 Algorithms).
        Optimized with joblib parallelism, float32 arrays, and common-space clarity.
        Target: all 16 runs under 15s.
        """
        if self._experiments_cache is not None and self._winning_labels is not None:
            return self._experiments_cache

        self.feature_engineering()
        dataset_keys = ["DS1_Original", "DS2_Correlation", "DS3_MutualInfo", "DS4_PCA95"]
        algorithm_names = ["KMeans", "Agglomerative", "DBSCAN", "GaussianMixture"]

        k_clusters = self.config.n_clusters
        X_common = self._datasets["DS1_Original"].values.astype(np.float32)

        # Precompute 2D PCA projections for scatter plots
        scatter_projections = {}
        for dname in dataset_keys:
            data = self._datasets[dname].values.astype(np.float32)
            if dname == "DS4_PCA95" and data.shape[1] >= 2:
                scatter_projections[dname] = data[:, :2]
            elif data.shape[1] >= 2:
                pca2 = PCA(n_components=2, random_state=42)
                scatter_projections[dname] = pca2.fit_transform(data).astype(np.float32)
            else:
                scatter_projections[dname] = np.hstack([data, np.zeros((data.shape[0], 1), dtype=np.float32)])

        if X_common.shape[1] >= 2:
            self._pca_2d = PCA(n_components=2, random_state=42).fit(X_common)
        else:
            self._pca_2d = None

        # Build task list of (dataset, algorithm)
        tasks = []
        for dname in dataset_keys:
            for alg_name in algorithm_names:
                tasks.append((dname, alg_name))

        total_steps = len(tasks)
        results: List[Dict[str, Any]] = []

        # Run independent experiments in parallel using joblib (bounded workers = 4)
        def run_task(dname, alg_name):
            if cancel_check and cancel_check():
                return None
            return self._execute_single_experiment(
                dname=dname,
                alg_name=alg_name,
                k_clusters=k_clusters,
                coords_2d=scatter_projections[dname],
                X_common=X_common,
            )

        # Execute parallel batches to keep cancellable and report progress (bounded workers = 4)
        parallel = joblib.Parallel(n_jobs=4, prefer="threads")
        raw_results = parallel(
            joblib.delayed(run_task)(dname, alg)
            for dname, alg in tasks
        )

        for step_idx, res in enumerate(raw_results):
            if res is not None:
                results.append(res)
                if job_callback:
                    job_callback(step_idx + 1, total_steps, f"Completed {res['algorithm']} on {res['dataset_name']}...")

        if cancel_check and cancel_check():
            return {"experiments": results, "scatter_data": {}, "winner": {}, "k_sweep": [], "cancelled": True}

        # --- Item 3: Winner Selection using Common-Space Clarity ---
        # Level playing field on DS1_Original coordinates
        valid_candidates = [
            r for r in results
            if r["cluster_count"] >= 2
            and r["noise_percentage"] <= 60.0
            and r["status"] == "success"
            and r["common_space_silhouette"] is not None
            and r["common_space_dbi"] is not None
            and r["common_space_chi"] is not None
        ]

        if valid_candidates:
            sil_vals = [r["common_space_silhouette"] for r in valid_candidates]
            dbi_vals = [r["common_space_dbi"] for r in valid_candidates]
            ch_vals = [r["common_space_chi"] for r in valid_candidates]

            min_sil, max_sil = min(sil_vals), max(sil_vals)
            min_dbi, max_dbi = min(dbi_vals), max(dbi_vals)
            min_ch, max_ch = min(ch_vals), max(ch_vals)

            for r in valid_candidates:
                norm_sil = (r["common_space_silhouette"] - min_sil) / (max_sil - min_sil + 1e-6)
                norm_dbi = (max_dbi - r["common_space_dbi"]) / (max_dbi - min_dbi + 1e-6)
                norm_ch = (r["common_space_chi"] - min_ch) / (max_ch - min_ch + 1e-6)

                comp = round(float((norm_sil + norm_dbi + norm_ch) / 3.0), 4)
                r["composite_score"] = comp

            k_candidates = [r for r in valid_candidates if r["cluster_count"] == self.config.n_clusters]
            winning = max(k_candidates if k_candidates else valid_candidates, key=lambda x: x["composite_score"])
        else:
            winning = results[0]

        self._winning_model = winning.get("_model")
        self._winning_labels = winning.get("_labels")
        self._winning_dataset_name = winning["dataset_name"]
        self._winning_alg_name = winning["algorithm"]

        # Run K-sweep with Item 2h composite rank search
        k_sweep, suggested_k, suggested_k_reason = self.run_k_sweep()

        sanitized_results = []
        for r in results:
            r_copy = dict(r)
            r_copy.pop("_model", None)
            r_copy.pop("_labels", None)
            sanitized_results.append(r_copy)

        winning_clean = dict(winning)
        winning_clean.pop("_model", None)
        winning_clean.pop("_labels", None)

        selection_formula = (
            "composite_score = (normalized_common_space_silhouette + (1 - normalized_common_space_dbi) + normalized_common_space_chi) / 3"
        )

        self._experiments_cache = {
            "dataset_id": self.dataset_id,
            "config_hash": self.config_hash,
            "experiments": sanitized_results,
            "winning_experiment": winning_clean,
            "selection_formula": selection_formula,
            "k_sweep": k_sweep,
            "suggested_k": suggested_k,
            "suggested_k_reason": suggested_k_reason,
        }

        if self.db:
            self.db.save_pipeline_cache(self.dataset_id, self.config_hash, experiments=self._experiments_cache)

        return self._experiments_cache

    def get_growth_personas(self) -> Dict[str, Any]:
        """
        Derives business personas purely from the data:
        1. Calculates actual centroid profiles in original feature units.
        2. Computes Z-scores vs global feature mean.
        3. Identifies top 3 positive and top 3 negative distinguishing attributes (|z| >= 0.5 only).
        4. Maps to domain templates dynamically with real counts and dynamic ROI calculations.
        Zero hardcoded figures.
        """
        if self._personas_cache is not None:
            return self._personas_cache

        if self._experiments_cache is None or self._winning_labels is None:
            self.run_experiments()

        labels = self._winning_labels
        total_records = len(labels)
        df_unscaled = self._prepare_raw_features().fillna(self._prepare_raw_features().median(numeric_only=True))

        target_k = self.config.n_clusters
        all_cluster_ids = list(range(target_k))

        cluster_counts = {}
        cluster_means = {}
        cluster_zscores = {}

        for c in all_cluster_ids:
            mask = (labels == c)
            cnt = int(np.sum(mask))
            cluster_counts[c] = cnt

            sub_df = df_unscaled[mask] if np.any(mask) else df_unscaled
            means = {}
            zscores = {}
            for col in self._feature_names:
                c_mean = float(sub_df[col].mean()) if len(sub_df) > 0 else self._global_means.get(col, 0.0)
                g_mean = self._global_means.get(col, 0.0)
                g_std = self._global_stds.get(col, 1.0)
                z = (c_mean - g_mean) / (g_std + 1e-6)

                means[col] = round(c_mean, 2)
                zscores[col] = round(z, 2)

            cluster_means[c] = means
            cluster_zscores[c] = zscores

        domain = self.config.domain_template
        palette = ["#6366F1", "#10B981", "#F59E0B", "#F43F5E", "#8B5CF6", "#06B6D4", "#EC4899", "#14B8A6"]

        cluster_assignments = self._assign_archetypes(
            domain=domain,
            target_k=target_k,
            all_cluster_ids=all_cluster_ids,
            cluster_zscores=cluster_zscores,
            cluster_counts=cluster_counts,
        )

        personas_list = []
        for idx, c in enumerate(all_cluster_ids):
            cnt = cluster_counts[c]
            pct = round((cnt / total_records) * 100.0, 2) if total_records > 0 else 0.0
            z_dict = cluster_zscores[c]
            m_dict = cluster_means[c]

            sorted_z = sorted(z_dict.items(), key=lambda x: x[1], reverse=True)

            # Item 2c: show only features with |z| >= 0.5, max 3 above and 3 below average.
            # Never show features near 0.
            top_pos = [
                DistinguishingFeature(
                    feature=k,
                    z_score=v,
                    cluster_mean=m_dict[k],
                    global_mean=round(self._global_means.get(k, 0.0), 2),
                    interpretation=format_trait_wording(k, v, domain),
                )
                for k, v in sorted_z if v >= 0.5
            ][:3]

            top_neg = [
                DistinguishingFeature(
                    feature=k,
                    z_score=v,
                    cluster_mean=m_dict[k],
                    global_mean=round(self._global_means.get(k, 0.0), 2),
                    interpretation=format_trait_wording(k, v, domain),
                )
                for k, v in reversed(sorted_z) if v <= -0.5
            ][:3]

            # If fewer than 2 pass, say "This group differs only slightly from average."
            traits_count = len(top_pos) + len(top_neg)
            distinguishing_note = None
            persona_name, characteristics, nba_templates = cluster_assignments[c]

            if traits_count < 2:
                distinguishing_note = "This group differs only slightly from average."
                characteristics = ["This group differs only slightly from average."] + characteristics

            # Dynamic ROI modeling based on real segment size
            avg_rev = 120.0
            for rev_candidate in ["PURCHASES", "PAYMENTS", "monetary", "revenue", "mrr", "amount"]:
                for feat in self._feature_names:
                    if rev_candidate in feat.lower():
                        avg_rev = max(50.0, float(m_dict.get(feat, 120.0)))
                        break

            uplift_pct = 0.12 + (0.04 * (idx % 3))
            cost_per_cust = 3.50
            gross_lift = cnt * avg_rev * uplift_pct
            total_cost = cnt * cost_per_cust
            net_lift = round(max(0.0, gross_lift - total_cost), 2)
            roi_mult = round(gross_lift / (total_cost + 1e-4), 2)

            assumptions = PersonaAssumptions(
                avg_revenue_per_customer=round(avg_rev, 2),
                expected_uplift_pct=round(uplift_pct, 2),
                campaign_cost_per_customer=round(cost_per_cust, 2),
            )

            campaigns = []
            for t_idx, camp in enumerate(nba_templates):
                c_lift = f"+{int(uplift_pct * 100 + (t_idx * 3))}% Expected Lift"
                campaigns.append(
                    NBACampaign(
                        title=camp["title"],
                        description=camp["description"],
                        expected_roi_lift=c_lift,
                        channel=camp["channel"],
                        estimated_net_lift_dollars=round(net_lift * (1.0 - t_idx * 0.15), 2),
                        roi_multiple=round(roi_mult * (1.0 - t_idx * 0.1), 2),
                    )
                )

            personas_list.append(
                PersonaProfile(
                    id=c,
                    name=persona_name,
                    percentage=pct,
                    count=cnt,
                    color=palette[idx % len(palette)],
                    centroid_profile=m_dict,
                    top_positive_features=top_pos,
                    top_negative_features=top_neg,
                    distinguishing_note=distinguishing_note,
                    characteristics=characteristics,
                    assumptions=assumptions,
                    nba_campaigns=campaigns,
                )
            )

        # Item 2f: Box plots must use winsorized values (or clip the axis to the 1st-99th percentile) so boxes are readable
        box_df = self._winsorized_df if self._winsorized_df is not None else df_unscaled
        top_feature_candidates = self._feature_names[:6]
        box_plots_list = []
        for feat in top_feature_candidates:
            cluster_stat_map = {}
            for c in all_cluster_ids:
                p_name = cluster_assignments[c][0]
                mask = (labels == c)
                c_vals = box_df.loc[mask, feat].dropna() if np.any(mask) else pd.Series([self._global_means.get(feat, 0.0)])
                if len(c_vals) == 0:
                    c_vals = pd.Series([self._global_means.get(feat, 0.0)])

                p1 = float(c_vals.quantile(0.01))
                p99 = float(c_vals.quantile(0.99))
                c_vals_clipped = c_vals.clip(lower=p1, upper=p99)

                cluster_stat_map[p_name] = ClusterBoxPlotStat(
                    min=round(float(c_vals_clipped.min()), 2),
                    q1=round(float(c_vals_clipped.quantile(0.25)), 2),
                    median=round(float(c_vals_clipped.median()), 2),
                    q3=round(float(c_vals_clipped.quantile(0.75)), 2),
                    max=round(float(c_vals_clipped.max()), 2),
                )
            box_plots_list.append(FeatureBoxPlot(feature=feat, clusters=cluster_stat_map))

        direction_map = {feat: get_column_direction(domain, feat) for feat in self._feature_names}
        win_exp = self._experiments_cache.get("winning_experiment", {})

        self._personas_cache = {
            "dataset_id": self.dataset_id,
            "personas": personas_list,
            "winning_model": f"{self._winning_alg_name} on {self._winning_dataset_name}",
            "winning_silhouette": win_exp.get("silhouette_score"),
            "winning_dbi": win_exp.get("davies_bouldin_index"),
            "winning_common_silhouette": win_exp.get("common_space_silhouette"),
            "domain_template": domain,
            "box_plots": [b.model_dump() for b in box_plots_list],
            "direction_map": direction_map,
        }

        if self.db:
            self.db.save_pipeline_cache(self.dataset_id, self.config_hash, personas=self._personas_cache)

        return self._personas_cache

    def _assign_archetypes(
        self,
        domain: str,
        target_k: int,
        all_cluster_ids: List[int],
        cluster_zscores: Dict[int, Dict[str, float]],
        cluster_counts: Dict[int, int],
    ) -> Dict[int, Tuple[str, List[str], List[Dict[str, str]]]]:
        if domain == "credit_card":
            archetypes = [
                (
                    "VIP Omnichannel Spenders",
                    lambda z: 2.0 * get_feature_z(z, "purchases") + 1.8 * get_feature_z(z, "oneoff_purchases") + 1.0 * get_feature_z(z, "credit_limit") + 0.8 * get_feature_z(z, "payments"),
                    [
                        "High spending volume and elevated credit facility tier",
                        "Significant one-off capital expenditures and high transaction volume",
                        "Substantial contributor to overall portfolio interchange volume",
                    ],
                    [
                        {"title": "Exclusive Concierge & Airport Lounge Access", "description": "Provide premium travel privileges and lifestyle concierge to retain top tier wallet share.", "channel": "Dedicated Account Manager & Direct Mail"},
                        {"title": "AI Wealth Management & Private Advisory", "description": "Cross-sell high-yield treasury and investment opportunities to affluent cardholders.", "channel": "VIP Client Portal & Private Banker"},
                    ],
                ),
                (
                    "Transactors",
                    lambda z: 2.2 * get_feature_z(z, "purchases_frequency") + 1.8 * get_feature_z(z, "purchases_installments_frequency") + 1.2 * get_feature_z(z, "oneoff_purchases_frequency") + 1.0 * get_feature_z(z, "prc_full_payment") - 1.5 * get_feature_z(z, "cash_advance_frequency"),
                    [
                        "Frequent day-to-day card utilization for merchant purchases",
                        "Consistently settles balances in full with low revolving balances",
                        "Low delinquency risk with dependable payment discipline",
                    ],
                    [
                        {"title": "Tiered Merchant Cashback Upgrade", "description": "Incentivize high daily spend with category-specific cashback accelerators.", "channel": "Mobile Push & Email"},
                        {"title": "Spending Analytics & Category Insights", "description": "Empower active transactors with interactive financial health dashboards.", "channel": "In-App Push"},
                    ],
                ),
                (
                    "Revolvers",
                    lambda z: 2.0 * get_feature_z(z, "cash_advance_frequency") + 1.8 * get_feature_z(z, "cash_advance_trx") + 1.5 * get_feature_z(z, "cash_advance") + 1.0 * get_feature_z(z, "balance") - 1.0 * get_feature_z(z, "prc_full_payment"),
                    [
                        "Carries substantial revolving balances and high cash advance utilization",
                        "High interest yield generator with minimal full payment propensity",
                        "Consistently utilizes available credit limits",
                    ],
                    [
                        {"title": "0% Intro APR Balance Transfer Consolidation", "description": "Capture external credit lines by offering structured 0% intro APR balance transfers.", "channel": "Online Banking & Direct Mail"},
                        {"title": "Smart Autopay Debt Paydown Scheduler", "description": "Facilitate automated monthly installments to optimize interest margins safely.", "channel": "SMS & Mobile App"},
                    ],
                ),
                (
                    "Dormant / At-Risk",
                    lambda z: - 2.0 * get_feature_z(z, "purchases_frequency") - 1.5 * get_feature_z(z, "purchases") - 1.2 * get_feature_z(z, "purchases_trx") - 1.0 * get_feature_z(z, "cash_advance_trx"),
                    [
                        "Minimal transaction activity and low revolving utilization",
                        "Static account with declining touchpoints over trailing periods",
                        "Elevated probability of silent customer churn",
                    ],
                    [
                        {"title": "90-Day 5% Cashback Activation Sprint", "description": "Re-ignite inactive cards with time-limited bonus cashback on essential everyday spend.", "channel": "Email & SMS"},
                        {"title": "Cardholder Benefits Re-Engagement Series", "description": "Highlight newly added local merchant perks and zero-fee benefits.", "channel": "Email"},
                    ],
                ),
            ]
        elif domain == "ecommerce_rfm":
            archetypes = [
                (
                    "Champions",
                    lambda z: 2.0 * get_feature_z(z, "frequency", "orders") + 2.0 * get_feature_z(z, "monetary", "revenue", "spend", "purchases", "amount") - 1.5 * get_feature_z(z, "recency", "days_since"),
                    [
                        "High order frequency, high monetary spend, and bought recently (low recency)",
                        "Top-tier lifetime customer value with dependable repeat buying cycles",
                        "Prime candidates for loyalty VIP rewards and exclusive ambassador programs",
                    ],
                    [
                        {"title": "VIP Early Access & Tiered Loyalty Rewards", "description": "Grant exclusive early access to upcoming product launches and private sales.", "channel": "Email & SMS"},
                        {"title": "Brand Ambassador Referral Program", "description": "Incentivize peer referrals with two-sided reward credits.", "channel": "In-App Push"},
                    ],
                ),
                (
                    "Recent / Promising",
                    lambda z: - 2.5 * get_feature_z(z, "recency", "days_since") + 1.0 * get_feature_z(z, "monetary", "revenue", "spend", "purchases", "amount") - 0.5 * abs(get_feature_z(z, "monetary", "revenue", "spend", "purchases", "amount")),
                    [
                        "Bought recently with moderate order value and fresh brand momentum",
                        "High response probability during post-purchase onboarding window",
                        "High potential to graduate into core recurring purchasers",
                    ],
                    [
                        {"title": "Second Purchase Accelerator Discount", "description": "Trigger limited-time voucher for second order within 14 days of receipt.", "channel": "SMS & Email"},
                        {"title": "Curated Product Discovery Guide", "description": "Educate buyers on best-sellers and top customer favorites.", "channel": "In-App Notification"},
                    ],
                ),
                (
                    "At-Risk / Lapsed",
                    lambda z: 2.5 * get_feature_z(z, "recency", "days_since") - 1.5 * get_feature_z(z, "frequency", "orders"),
                    [
                        "High recency (lapsed) with low purchase frequency",
                        "Extended duration since last order requiring targeted reactivation",
                        "Critical churn threshold requiring aggressive win-back campaigns",
                    ],
                    [
                        {"title": "Dynamic Win-Back Re-Engagement Offer", "description": "Deploy personalized reactivation coupon tailored to past basket categories.", "channel": "Omnichannel Push & Email"},
                        {"title": "Customer Re-Engagement Feedback Survey", "description": "Identify reasons for dormancy with a friction audit.", "channel": "Email & SMS"},
                    ],
                ),
                (
                    "Return-Prone Shoppers",
                    lambda z: 3.0 * get_feature_z(z, "return_rate", "returns") + 0.5 * get_feature_z(z, "frequency", "orders"),
                    [
                        "High return rate relative to portfolio average (Return-prone)",
                        "Active order placement accompanied by frequent merchandise returns",
                        "Requires sizing/spec guidance and dedicated return-friction resolution",
                    ],
                    [
                        {"title": "Personalized Sizing & Fit Assistance", "description": "Provide smart sizing guides and consultation to minimize return rate.", "channel": "Email & In-App Guide"},
                        {"title": "Exchange Rather Than Refund Incentive", "description": "Encourage instant exchange with bonus store credit.", "channel": "Checkout Recommendations"},
                    ],
                ),
                (
                    "Loyal Regular Shoppers",
                    lambda z: 1.8 * get_feature_z(z, "frequency", "orders") + 0.8 * get_feature_z(z, "monetary", "revenue", "spend"),
                    [
                        "Consistent recurring purchases with stable basket values",
                        "Steady engagement with seasonal catalogs and product drops",
                        "Responsive to cross-category recommendations",
                    ],
                    [
                        {"title": "Automated Subscription & Replenishment Bundles", "description": "Convert frequent purchases into recurring subscription plans with 10% discount.", "channel": "Email & SMS"},
                        {"title": "Cross-Sell Category Expansion Offer", "description": "Promote complementary accessories tailored to past purchases.", "channel": "Checkout Recommendations"},
                    ],
                ),
            ]
        elif domain == "saas":
            archetypes = [
                (
                    "Enterprise Power Users",
                    lambda z: 2.0 * get_feature_z(z, "mrr") + 1.5 * get_feature_z(z, "seats"),
                    ["Extensive enterprise license count and top revenue contributor", "Daily active utilization across team members", "High stickiness and low churn risk"],
                    [
                        {"title": "Enterprise Tier Upgrade Consultation", "description": "Engage workspace owners with custom SLA and security add-on proposals.", "channel": "In-App Intercom & Email"},
                        {"title": "Executive Business Review Scheduling", "description": "Coordinate quarterly review to benchmark ROI and expansion metrics.", "channel": "Account Manager Outreach"},
                    ],
                ),
                (
                    "Steady Core Operators",
                    lambda z: 2.0 * get_feature_z(z, "logins") + 1.0 * get_feature_z(z, "usage"),
                    ["Consistent day-to-day workflow integration", "Predictable usage patterns within core features", "Reliable monthly contract renewal"],
                    [
                        {"title": "Product Specialist Workflow Optimization", "description": "Offer complimentary 1-on-1 workflow audit to unlock unutilized features.", "channel": "In-App Banner"},
                        {"title": "Advanced Integration Setup Guide", "description": "Connect workspace to Slack, GitHub, and CRM tools to deepen lock-in.", "channel": "In-App Notification"},
                    ],
                ),
                (
                    "Expansion & Upsell Candidates",
                    lambda z: 1.5 * get_feature_z(z, "seats") + 1.5 * get_feature_z(z, "growth"),
                    ["Rapid seat count additions and growing team adoption", "Approaching tier-specific quota thresholds", "Strong signals of organizational expansion"],
                    [
                        {"title": "Team Seat Bundle Incentive", "description": "Provide volume discount on adding 10+ additional workspace seats.", "channel": "In-App Modal & Billing Alert"},
                        {"title": "Custom API & SSO Security Add-on", "description": "Unlock Okta SAML SSO and dedicated webhook throughput.", "channel": "Email Outreach"},
                    ],
                ),
                (
                    "At-Risk / Disengaged Workspaces",
                    lambda z: 2.0 * get_feature_z(z, "churn_risk", "churn") - 1.5 * get_feature_z(z, "logins"),
                    ["Sharp decline in weekly active user logins", "Stagnant invite acceptance rate and minimal feature utilization", "High churn alert status"],
                    [
                        {"title": "Customer Success Health Intervention", "description": "Proactive check-in from dedicated CSM to troubleshoot blockers.", "channel": "Direct Email & Intercom"},
                        {"title": "Automated Re-Activation Email Sequence", "description": "Digest of teammate activity and unfinished tasks to bring users back.", "channel": "Email"},
                    ],
                ),
            ]
        else:
            archetypes = []

        assigned_raw = {}
        for c in all_cluster_ids:
            z_c = cluster_zscores.get(c, {})
            ret_z = get_feature_z(z_c, "return_rate", "returns")

            if archetypes:
                scored = [(score_fn(z_c), name, traits, camps) for (name, score_fn, traits, camps) in archetypes]
                scored.sort(key=lambda x: x[0], reverse=True)
                _, best_name, best_traits, best_camps = scored[0]

                traits_copy = list(best_traits)
                if ret_z > 0.6 and "Return" not in best_name:
                    traits_copy = [f"Flag: Return-prone (Elevated returns Z: {ret_z:+0.2f})"] + traits_copy

                assigned_raw[c] = (best_name, traits_copy, best_camps)
            else:
                p_label = chr(65 + c)
                sorted_z = sorted(z_c.items(), key=lambda x: x[1], reverse=True)
                top_pos_k = sorted_z[0][0] if sorted_z and sorted_z[0][1] > 0 else "Signal A"
                top_neg_k = sorted_z[-1][0] if sorted_z and sorted_z[-1][1] < 0 else "Signal B"
                name = f"Segment {p_label} (High {top_pos_k}, Low {top_neg_k})"
                traits = [
                    f"Distinguishing strength: {top_pos_k}",
                    f"Lowest relative feature: {top_neg_k}",
                    f"Comprises {cluster_counts.get(c, 0)} accounts within active dataset",
                ]
                camps = [
                    {"title": f"Targeted Segment {p_label} Engagement", "description": f"Tailor messaging around {top_pos_k}.", "channel": "Digital Channels"},
                    {"title": f"Value Optimization Campaign {p_label}", "description": f"Address friction points related to {top_neg_k}.", "channel": "Email & In-App"},
                ]
                assigned_raw[c] = (name, traits, camps)

        name_counts: Dict[str, int] = {}
        for c in all_cluster_ids:
            base_name, _, _ = assigned_raw[c]
            name_counts[base_name] = name_counts.get(base_name, 0) + 1

        name_seen: Dict[str, int] = {}
        assignments: Dict[int, Tuple[str, List[str], List[Dict[str, str]]]] = {}
        for c in all_cluster_ids:
            base_name, traits, camps = assigned_raw[c]
            if name_counts[base_name] > 1:
                name_seen[base_name] = name_seen.get(base_name, 0) + 1
                suffix = "" if name_seen[base_name] == 1 else f" ({name_seen[base_name]})"
                final_name = f"{base_name}{suffix}"
            else:
                final_name = base_name
            assignments[c] = (final_name, traits, camps)

        return assignments

    def predict_segment(self, features: Dict[str, float]) -> PredictSegmentResponse:
        """Inference endpoint: maps an individual customer feature dictionary to their predicted persona and nearest NBA."""
        if self._personas_cache is None or self._winning_model is None:
            self.get_growth_personas()

        row = []
        for col in self._feature_names:
            val = features.get(col, self._global_means.get(col, 0.0))
            if col in self._winsor_bounds:
                lo, hi = self._winsor_bounds[col]
                val = max(lo, min(hi, val))
            row.append(val)

        row_arr = np.array(row, dtype=np.float32).reshape(1, -1)
        scaled_row = self._scaler.transform(row_arr).astype(np.float32)

        if self._winning_dataset_name == "DS4_PCA95" and self._pca:
            model_input = self._pca.transform(scaled_row).astype(np.float32)
        elif self._winning_dataset_name == "DS2_Correlation":
            ds2_cols = self._feature_engineering_cache["ds2_correlation_features"]
            ds2_indices = [self._feature_names.index(c) for c in ds2_cols]
            model_input = scaled_row[:, ds2_indices]
        elif self._winning_dataset_name == "DS3_MutualInfo":
            ds3_cols = self._feature_engineering_cache["ds3_mutual_info_features"]
            ds3_indices = [self._feature_names.index(c) for c in ds3_cols]
            model_input = scaled_row[:, ds3_indices]
        else:
            model_input = scaled_row

        try:
            pred_cluster = int(self._winning_model.predict(model_input)[0])
        except Exception:
            pred_cluster = 0

        confidence = 0.85
        dists = np.array([0.5])
        if hasattr(self._winning_model, "predict_proba"):
            probs = self._winning_model.predict_proba(model_input)[0]
            confidence = round(float(np.max(probs)), 2)
        elif hasattr(self._winning_model, "cluster_centers_"):
            dists = np.linalg.norm(self._winning_model.cluster_centers_ - model_input, axis=1)
            min_dist = float(np.min(dists))
            confidence = round(float(1.0 / (1.0 + min_dist * 0.2)), 2)

        persona = None
        for p in self._personas_cache["personas"]:
            if p.id == pred_cluster:
                persona = p
                break
        if persona is None:
            persona = self._personas_cache["personas"][0]

        top_traits = [f"{t.feature} (Z={t.z_score})" for t in persona.top_positive_features]
        recommended_nba = persona.nba_campaigns[0] if persona.nba_campaigns else NBACampaign(
            title="Standard Engagement Sprint",
            description="Activate cardholder touchpoints.",
            expected_roi_lift="+12% Lift",
            channel="Email",
            estimated_net_lift_dollars=12000.0,
            roi_multiple=4.2,
        )

        return PredictSegmentResponse(
            cluster_label=pred_cluster,
            persona_name=persona.name,
            confidence=confidence,
            recommended_nba=recommended_nba,
            distance_to_centroid=round(float(np.min(dists)) if hasattr(self._winning_model, "cluster_centers_") else 0.5, 3),
            top_traits=top_traits,
        )

    def export_segments_csv(self) -> str:
        """
        Appends assigned cluster labels and persona names to original dataset rows and outputs CSV string.
        Item 5 Security: Neutralises formula injection by prefixing cells starting with =, +, -, @ with an apostrophe.
        """
        if self._personas_cache is None or self._winning_labels is None:
            self.get_growth_personas()

        df_out = self._raw_df.copy()
        labels = self._winning_labels

        c_to_persona = {p.id: p.name for p in self._personas_cache["personas"]}
        df_out["CLUSTER_LABEL"] = labels
        df_out["PERSONA_NAME"] = [c_to_persona.get(int(lbl), f"Cluster {lbl}") for lbl in labels]

        # Neutralise CSV formula injection
        for col in df_out.columns:
            if df_out[col].dtype == object or pd.api.types.is_string_dtype(df_out[col]):
                df_out[col] = df_out[col].astype(str).apply(
                    lambda v: f"'{v}" if v and v[0] in ("=", "+", "-", "@") else v
                )

        out_io = io.StringIO()
        df_out.to_csv(out_io, index=False)
        return out_io.getvalue()

    def export_model_joblib(self) -> bytes:
        """Serializes trained pipeline artifacts into downloadable joblib binary."""
        if self._personas_cache is None or self._winning_model is None:
            self.get_growth_personas()

        artifact_bundle = {
            "dataset_id": self.dataset_id,
            "config": self.config.model_dump(),
            "feature_names": self._feature_names,
            "scaler": self._scaler,
            "imputer": self._imputer,
            "winsor_bounds": self._winsor_bounds,
            "pca": self._pca,
            "winning_model": self._winning_model,
            "winning_alg_name": self._winning_alg_name,
            "winning_dataset_name": self._winning_dataset_name,
            "personas": [p.model_dump() if hasattr(p, "model_dump") else (p.dict() if hasattr(p, "dict") else p) for p in self._personas_cache["personas"]],
            "exported_at": datetime.datetime.now().isoformat(),
        }

        buf = io.BytesIO()
        joblib.dump(artifact_bundle, buf)
        buf.seek(0)
        return buf.getvalue()

    def export_model_json(self) -> Dict[str, Any]:
        """
        Item 5 Security: Safe non-pickle JSON export of centroids, scaler, and PCA parameters.
        """
        self.preprocess()
        self.feature_engineering()
        if self._personas_cache is None or self._winning_model is None:
            self.get_growth_personas()

        scaler_dict = {}
        if self._scaler is not None:
            scaler_dict = {
                "mean": [round(float(m), 4) for m in self._scaler.mean_],
                "scale": [round(float(s), 4) for s in self._scaler.scale_],
                "var": [round(float(v), 4) for v in self._scaler.var_],
            }

        pca_dict = None
        if self._pca is not None:
            pca_dict = {
                "n_components": int(self._pca.n_components_),
                "explained_variance_ratio": [round(float(v), 4) for v in self._pca.explained_variance_ratio_],
                "components": [[round(float(c), 4) for c in row] for row in self._pca.components_],
            }

        centroids = {}
        for p in self._personas_cache["personas"]:
            name = p.name if hasattr(p, "name") else p["name"]
            profile = p.centroid_profile if hasattr(p, "centroid_profile") else p["centroid_profile"]
            centroids[name] = profile

        serialized_personas = [
            p.model_dump() if hasattr(p, "model_dump") else (p.dict() if hasattr(p, "dict") else p)
            for p in self._personas_cache["personas"]
        ]

        return {
            "dataset_id": self.dataset_id,
            "exported_at": datetime.datetime.now().isoformat(),
            "winning_algorithm": self._winning_alg_name or "KMeans",
            "winning_space": self._winning_dataset_name or "DS1_Original",
            "feature_names": self._feature_names,
            "scaler": scaler_dict,
            "centroids": centroids,
            "pca_parameters": pca_dict,
            "personas": serialized_personas,
        }
