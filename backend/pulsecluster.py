"""
PulseCluster: Adaptive Consensus Segmentation Algorithm
======================================================
An ensemble clustering method constructed from well-established machine learning
components. It is NOT a newly invented fundamental clustering equation, but rather
a robust engineering consensus pipeline that combines:
  1. Smart distribution-aware preprocessing (KNN imputation, Yeo-Johnson / log1p skew correction,
     percentile winsorization, Z-score standardization).
  2. Optional domain-aware financial/behavioral ratio engineering.
  3. PCA dimensionality reduction with whitening.
  4. Parallel bootstrap subsampling of K-Means and Gaussian Mixture Models (GMM) across candidate k (3..8).
  5. Maximum bipartite matching (Hungarian algorithm) cluster alignment to establish majority consensus.
  6. Final centroid-seeded K-Means refinement over the full dataset.
  7. Adjusted Rand Index (ARI) bootstrap stability measurement (reliability score 0-100).
  8. Composite ranking over common-space silhouette, Calinski-Harabasz, inverse Davies-Bouldin,
     and bootstrap stability, enforcing a >= 3% minimum cluster size penalty.
  9. Centroid margin customer confidence scoring (0-100).
  10. Strict execution time budgeting with fine-grained timing breakdowns.
"""

import time
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import linear_sum_assignment
from scipy.stats import skew
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
import joblib


class PulseCluster(BaseEstimator, ClusterMixin):
    """
    Adaptive Consensus Segmentation algorithm.
    Ensemble method built from known statistical and machine learning components.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        k_min: int = 3,
        k_max: int = 8,
        n_bootstraps: int = 10,
        subsample_ratio: float = 0.8,
        max_search_rows: int = 10000,
        time_budget_sec: float = 20.0,
        pca_variance: float = 0.95,
        pca_whiten: bool = True,
        engineer_ratios: bool = True,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.k_min = k_min
        self.k_max = k_max
        self.n_bootstraps = n_bootstraps
        self.subsample_ratio = subsample_ratio
        self.max_search_rows = max_search_rows
        self.time_budget_sec = time_budget_sec
        self.pca_variance = pca_variance
        self.pca_whiten = pca_whiten
        self.engineer_ratios = engineer_ratios
        self.random_state = random_state

        # Fitted artifacts
        self.imputer_: Optional[KNNImputer] = None
        self.skew_transforms_: Dict[str, Any] = {}
        self.winsor_bounds_: Dict[str, Tuple[float, float]] = {}
        self.scaler_: Optional[StandardScaler] = None
        self.pca_: Optional[PCA] = None
        self.final_kmeans_: Optional[KMeans] = None
        self.feature_names_in_: List[str] = []
        self.raw_feature_names_in_: List[str] = []
        
        # Results and metrics
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None  # in PCA space
        self.cluster_centers_original_: Optional[np.ndarray] = None  # in original feature space
        self.selected_k_: int = 4
        self.selected_base_method_: str = "KMeans"
        self.stability_score_: float = 0.0  # Mean ARI
        self.reliability_: float = 0.0  # stability * 100 (0..100)
        self.timing_breakdown_: Dict[str, float] = {}
        self.candidate_scores_: List[Dict[str, Any]] = []

    def _engineer_domain_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers standard behavioral/financial ratios if relevant columns exist:
          - credit utilization = BALANCE / CREDIT_LIMIT
          - payment ratio = PAYMENTS / MINIMUM_PAYMENTS
          - average purchase = PURCHASES / PURCHASES_TRX
        Guards strictly against zero division.
        """
        df_out = df.copy()
        cols_upper = {c.upper(): c for c in df_out.columns}

        # 1. Credit Utilization
        if "BALANCE" in cols_upper and "CREDIT_LIMIT" in cols_upper:
            bal_col = cols_upper["BALANCE"]
            lim_col = cols_upper["CREDIT_LIMIT"]
            denom = df_out[lim_col].replace(0, np.nan).fillna(1.0)
            df_out["CREDIT_UTILIZATION"] = (df_out[bal_col] / denom).clip(0, 10.0)

        # 2. Payment Ratio
        if "PAYMENTS" in cols_upper and "MINIMUM_PAYMENTS" in cols_upper:
            pay_col = cols_upper["PAYMENTS"]
            min_col = cols_upper["MINIMUM_PAYMENTS"]
            denom = df_out[min_col].replace(0, np.nan).fillna(1.0)
            df_out["PAYMENT_RATIO"] = (df_out[pay_col] / denom).clip(0, 50.0)

        # 3. Average Purchase
        if "PURCHASES" in cols_upper and "PURCHASES_TRX" in cols_upper:
            pur_col = cols_upper["PURCHASES"]
            trx_col = cols_upper["PURCHASES_TRX"]
            denom = df_out[trx_col].replace(0, np.nan).fillna(1.0)
            df_out["AVG_PURCHASE_TRX"] = (df_out[pur_col] / denom).clip(0, 10000.0)

        return df_out

    def _smart_preprocess_fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Step 1 Smart Preprocessing:
          1. KNN Impute (k=5)
          2. For features with |skew| > 1, apply Yeo-Johnson (or log1p if non-negative)
          3. Winsorize at 1st and 99th percentiles
          4. StandardScaler
        """
        t0 = time.perf_counter()
        self.raw_feature_names_in_ = list(df.columns)
        working_df = df.copy()

        if self.engineer_ratios:
            working_df = self._engineer_domain_ratios(working_df)

        self.feature_names_in_ = list(working_df.columns)
        n_samples = len(working_df)

        # 1. KNN Imputation (k=5)
        has_na = working_df.isna().any().any()
        if has_na:
            k_imp = min(5, max(1, n_samples - 1))
            self.imputer_ = KNNImputer(n_neighbors=k_imp)
            if n_samples > 3000:
                sample_train = working_df.dropna()
                if len(sample_train) < k_imp:
                    sample_train = working_df.sample(n=min(3000, n_samples), random_state=self.random_state)
                else:
                    sample_train = sample_train.sample(n=min(3000, len(sample_train)), random_state=self.random_state)
                self.imputer_.fit(sample_train)
                imputed_arr = self.imputer_.transform(working_df)
            else:
                imputed_arr = self.imputer_.fit_transform(working_df)
            clean_df = pd.DataFrame(imputed_arr, columns=self.feature_names_in_, index=working_df.index)
        else:
            self.imputer_ = None
            clean_df = working_df.copy()

        # 2. Skew correction: |skew| > 1 -> Yeo-Johnson or log1p
        self.skew_transforms_ = {}
        transformed_df = clean_df.copy()
        for col in self.feature_names_in_:
            col_vals = transformed_df[col].values
            col_skew = float(skew(col_vals, nan_policy="omit"))
            if abs(col_skew) > 1.0:
                min_val = np.nanmin(col_vals)
                if min_val >= 0:
                    # Non-negative: log1p is fast and perfectly invertible/safe
                    transformed_df[col] = np.log1p(col_vals)
                    self.skew_transforms_[col] = "log1p"
                else:
                    try:
                        pt = PowerTransformer(method="yeo-johnson", standardize=False)
                        transformed_df[col] = pt.fit_transform(col_vals.reshape(-1, 1)).flatten()
                        self.skew_transforms_[col] = pt
                    except Exception:
                        self.skew_transforms_[col] = "identity"
            else:
                self.skew_transforms_[col] = "identity"

        # 3. Percentile Winsorization (1st and 99th percentile)
        self.winsor_bounds_ = {}
        winsor_df = transformed_df.copy()
        for col in self.feature_names_in_:
            p1 = float(np.percentile(winsor_df[col], 1))
            p99 = float(np.percentile(winsor_df[col], 99))
            if p99 > p1:
                self.winsor_bounds_[col] = (p1, p99)
                winsor_df[col] = np.clip(winsor_df[col], p1, p99)
            else:
                self.winsor_bounds_[col] = (p1 - 1e-6, p99 + 1e-6)

        # 4. StandardScaler Z-score normalization
        self.scaler_ = StandardScaler()
        scaled_arr = self.scaler_.fit_transform(winsor_df).astype(np.float32)

        self.timing_breakdown_["preprocessing_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return scaled_arr, self.feature_names_in_

    def _transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """Applies fitted preprocessing and PCA to unseen data."""
        working_df = df.copy()
        if self.engineer_ratios:
            working_df = self._engineer_domain_ratios(working_df)

        # Align columns
        for col in self.feature_names_in_:
            if col not in working_df.columns:
                working_df[col] = 0.0
        working_df = working_df[self.feature_names_in_]

        if self.imputer_ is not None:
            imputed = self.imputer_.transform(working_df)
            clean_df = pd.DataFrame(imputed, columns=self.feature_names_in_, index=working_df.index)
        else:
            clean_df = working_df.fillna(0.0)

        transformed_df = clean_df.copy()
        for col in self.feature_names_in_:
            tx = self.skew_transforms_.get(col, "identity")
            if tx == "log1p":
                vals = np.maximum(0.0, transformed_df[col].values)
                transformed_df[col] = np.log1p(vals)
            elif isinstance(tx, PowerTransformer):
                try:
                    transformed_df[col] = tx.transform(transformed_df[col].values.reshape(-1, 1)).flatten()
                except Exception:
                    pass

        winsor_df = transformed_df.copy()
        for col in self.feature_names_in_:
            if col in self.winsor_bounds_:
                lo, hi = self.winsor_bounds_[col]
                winsor_df[col] = np.clip(winsor_df[col], lo, hi)

        scaled = self.scaler_.transform(winsor_df).astype(np.float32)
        if self.pca_ is not None:
            return self.pca_.transform(scaled).astype(np.float32)
        return scaled

    @staticmethod
    def align_labels_hungarian(ref_labels: np.ndarray, target_labels: np.ndarray, k: int) -> np.ndarray:
        """
        Aligns target_labels to ref_labels using the Hungarian algorithm (linear_sum_assignment)
        to maximize label overlap.
        """
        cost_matrix = np.zeros((k, k), dtype=np.int32)
        for i in range(len(ref_labels)):
            r = int(ref_labels[i])
            t = int(target_labels[i])
            if 0 <= r < k and 0 <= t < k:
                cost_matrix[r, t] += 1

        # Maximize overlap = minimize negative overlap
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        mapping = {col: row for row, col in zip(row_ind, col_ind)}
        
        aligned = np.array([mapping.get(int(lbl), int(lbl)) for lbl in target_labels], dtype=int)
        return aligned

    def fit(self, X_input: Any, y: Any = None):
        """Fits the PulseCluster adaptive consensus segmentation model."""
        start_time = time.perf_counter()
        time_budget = self.time_budget_sec

        # 1. Data conversion
        if isinstance(X_input, pd.DataFrame):
            df_raw = X_input.copy()
        else:
            df_raw = pd.DataFrame(X_input, columns=[f"feat_{i}" for i in range(X_input.shape[1])])

        n_rows = len(df_raw)

        # 2. Smart Preprocessing
        X_scaled, _ = self._smart_preprocess_fit_transform(df_raw)

        # 3. PCA Whitening to 90-95% variance
        t_pca0 = time.perf_counter()
        target_var = min(0.99, max(0.50, self.pca_variance))
        n_features = X_scaled.shape[1]
        
        if n_features >= 2:
            self.pca_ = PCA(n_components=target_var, whiten=self.pca_whiten, random_state=self.random_state)
            try:
                X_pca = self.pca_.fit_transform(X_scaled).astype(np.float32)
                if X_pca.shape[1] < 2:
                    self.pca_ = PCA(n_components=min(2, n_features), whiten=self.pca_whiten, random_state=self.random_state)
                    X_pca = self.pca_.fit_transform(X_scaled).astype(np.float32)
            except Exception:
                self.pca_ = PCA(n_components=min(2, n_features), whiten=self.pca_whiten, random_state=self.random_state)
                X_pca = self.pca_.fit_transform(X_scaled).astype(np.float32)
        else:
            self.pca_ = None
            X_pca = X_scaled

        self.timing_breakdown_["pca_ms"] = round((time.perf_counter() - t_pca0) * 1000.0, 2)

        # 4. Search sample: at most max_search_rows (10,000)
        t_search0 = time.perf_counter()
        if n_rows > self.max_search_rows:
            rng_sample = np.random.RandomState(self.random_state)
            search_indices = rng_sample.choice(n_rows, size=self.max_search_rows, replace=False)
            X_search = X_pca[search_indices]
            X_scaled_search = X_scaled[search_indices]
        else:
            search_indices = np.arange(n_rows)
            X_search = X_pca
            X_scaled_search = X_scaled

        n_search = len(X_search)
        if n_search < 10:
            subsample_size = max(1, int(n_search * self.subsample_ratio))
            if subsample_size == 0:
                subsample_size = n_search
        else:
            subsample_size = min(n_search, max(4, int(n_search * self.subsample_ratio)))

        # Candidate k search range
        if self.n_clusters is not None:
            k_candidates = [min(self.n_clusters, max(1, n_search - 1))]
        else:
            k_candidates = [k for k in range(self.k_min, self.k_max + 1) if k < n_search]

        candidate_evaluations: List[Dict[str, Any]] = []

        # Candidate generation & consensus evaluation
        for k in k_candidates:
            # Check elapsed time budget
            elapsed = time.perf_counter() - start_time
            if elapsed >= time_budget * 0.75 and len(candidate_evaluations) > 0:
                break

            for base_method in ["KMeans", "GMM"]:
                bootstrap_runs_labels = []
                bootstrap_sub_indices = []

                # Run B bootstrap fits
                for b in range(self.n_bootstraps):
                    sub_seed = self.random_state + b * 13 + k * 101
                    rng_b = np.random.RandomState(sub_seed)
                    sub_idx = rng_b.choice(n_search, size=subsample_size, replace=False)
                    X_sub = X_search[sub_idx]

                    try:
                        if base_method == "KMeans":
                            km = KMeans(n_clusters=k, n_init=10, random_state=sub_seed)
                            sub_labels = km.fit_predict(X_sub)
                        else:
                            gmm = GaussianMixture(
                                n_components=k,
                                covariance_type="diag",
                                reg_covar=1e-3,
                                max_iter=25,
                                n_init=2,
                                random_state=sub_seed,
                            )
                            sub_labels = gmm.fit_predict(X_sub)
                        bootstrap_runs_labels.append(sub_labels)
                        bootstrap_sub_indices.append(sub_idx)
                    except Exception:
                        continue

                if len(bootstrap_runs_labels) < 2:
                    continue

                # Measure stability: pairwise adjusted Rand index across overlapping points
                pairwise_aris = []
                n_b_actual = len(bootstrap_runs_labels)
                for i in range(min(5, n_b_actual)):
                    for j in range(i + 1, min(5, n_b_actual)):
                        idx_i = bootstrap_sub_indices[i]
                        idx_j = bootstrap_sub_indices[j]
                        common_points, comm_i, comm_j = np.intersect1d(idx_i, idx_j, return_indices=True)
                        if len(common_points) >= k * 3:
                            ari = adjusted_rand_score(
                                bootstrap_runs_labels[i][comm_i],
                                bootstrap_runs_labels[j][comm_j],
                            )
                            pairwise_aris.append(ari)

                stability = float(np.mean(pairwise_aris)) if pairwise_aris else 0.5
                stability = max(0.0, min(1.0, stability))

                # Consensus via Hungarian alignment to reference run 0
                ref_sub_idx = bootstrap_sub_indices[0]
                ref_labels = bootstrap_runs_labels[0]
                aligned_full_votes = np.zeros((n_search, k), dtype=np.float32)

                # Reference run votes
                for local_i, global_i in enumerate(ref_sub_idx):
                    aligned_full_votes[global_i, ref_labels[local_i]] += 1.0

                for b in range(1, n_b_actual):
                    b_sub_idx = bootstrap_sub_indices[b]
                    b_labels = bootstrap_runs_labels[b]
                    common_pts, comm_ref, comm_b = np.intersect1d(ref_sub_idx, b_sub_idx, return_indices=True)
                    if len(common_pts) >= k:
                        aligned_b = self.align_labels_hungarian(ref_labels[comm_ref], b_labels[comm_b], k)
                        # Build mapping for whole b
                        # Quick nearest-neighbor/mapping
                        cost = np.zeros((k, k), dtype=np.int32)
                        for cr, cb in zip(ref_labels[comm_ref], b_labels[comm_b]):
                            cost[int(cr), int(cb)] += 1
                        r_ind, c_ind = linear_sum_assignment(-cost)
                        map_b = {c: r for r, c in zip(r_ind, c_ind)}
                        full_aligned_b = np.array([map_b.get(int(l), int(l)) for l in b_labels], dtype=int)
                    else:
                        full_aligned_b = b_labels

                    for local_i, global_i in enumerate(b_sub_idx):
                        c_lbl = full_aligned_b[local_i]
                        if 0 <= c_lbl < k:
                            aligned_full_votes[global_i, c_lbl] += 1.0

                consensus_search_labels = np.argmax(aligned_full_votes, axis=1)

                # Calculate consensus centroids on X_search
                consensus_centroids = np.zeros((k, X_search.shape[1]), dtype=np.float32)
                for c in range(k):
                    c_mask = (consensus_search_labels == c)
                    if np.sum(c_mask) > 0:
                        consensus_centroids[c] = np.mean(X_search[c_mask], axis=0)
                    else:
                        consensus_centroids[c] = X_search[np.random.randint(n_search)]

                # Check cluster sizes on search sample
                cluster_counts = [int(np.sum(consensus_search_labels == c)) for c in range(k)]
                min_cluster_pct = (min(cluster_counts) / n_search) * 100.0

                # Compute common-space metrics on X_scaled_search
                sample_eval_size = min(3000, n_search)
                if n_search > sample_eval_size:
                    eval_idx = np.random.RandomState(42).choice(n_search, size=sample_eval_size, replace=False)
                    X_eval = X_scaled_search[eval_idx]
                    l_eval = consensus_search_labels[eval_idx]
                else:
                    X_eval = X_scaled_search
                    l_eval = consensus_search_labels

                try:
                    sil = float(silhouette_score(X_eval, l_eval))
                    dbi = float(davies_bouldin_score(X_eval, l_eval))
                    chi = float(calinski_harabasz_score(X_eval, l_eval))
                except Exception:
                    sil, dbi, chi = 0.0, 5.0, 1.0

                candidate_evaluations.append({
                    "k": k,
                    "base_method": base_method,
                    "stability": round(stability, 4),
                    "reliability": round(stability * 100.0, 2),
                    "silhouette": round(sil, 4),
                    "dbi": round(dbi, 4),
                    "chi": round(chi, 2),
                    "min_cluster_pct": round(min_cluster_pct, 2),
                    "consensus_centroids": consensus_centroids,
                })

        self.timing_breakdown_["candidate_generation_ms"] = round((time.perf_counter() - t_search0) * 1000.0, 2)
        self.candidate_scores_ = candidate_evaluations

        # 7. Composite Ranking & Selection
        t_refine0 = time.perf_counter()
        if not candidate_evaluations:
            # Fallback safe option
            best_cand = {
                "k": self.n_clusters or 4,
                "base_method": "KMeans",
                "stability": 0.5,
                "reliability": 50.0,
                "silhouette": 0.2,
                "dbi": 1.5,
                "chi": 500.0,
                "min_cluster_pct": 20.0,
                "consensus_centroids": "k-means++",
            }
        else:
            # Min-max normalize across candidates
            sil_vals = [c["silhouette"] for c in candidate_evaluations]
            dbi_vals = [c["dbi"] for c in candidate_evaluations]
            chi_vals = [c["chi"] for c in candidate_evaluations]
            stab_vals = [c["stability"] for c in candidate_evaluations]

            min_sil, max_sil = min(sil_vals), max(sil_vals)
            min_dbi, max_dbi = min(dbi_vals), max(dbi_vals)
            min_chi, max_chi = min(chi_vals), max(chi_vals)
            min_st, max_st = min(stab_vals), max(stab_vals)

            for cand in candidate_evaluations:
                norm_sil = (cand["silhouette"] - min_sil) / (max_sil - min_sil + 1e-6)
                norm_dbi = (max_dbi - cand["dbi"]) / (max_dbi - min_dbi + 1e-6)
                norm_chi = (cand["chi"] - min_chi) / (max_chi - min_chi + 1e-6)
                norm_stab = (cand["stability"] - min_st) / (max_st - min_st + 1e-6)

                raw_score = (norm_sil + norm_dbi + norm_chi + norm_stab) / 4.0
                
                # Penalty for any cluster under 3% of rows
                if cand["min_cluster_pct"] < 3.0:
                    raw_score -= 0.50

                cand["composite_score"] = round(float(raw_score), 4)

            best_cand = max(candidate_evaluations, key=lambda c: c["composite_score"])

        self.selected_k_ = best_cand["k"]
        self.selected_base_method_ = best_cand["base_method"]
        self.stability_score_ = best_cand["stability"]
        self.reliability_ = best_cand["reliability"]

        # 5. Consensus Centroids refinement with final KMeans on the FULL data
        k_final = self.selected_k_
        init_centers = best_cand.get("consensus_centroids")
        if isinstance(init_centers, np.ndarray) and init_centers.shape == (k_final, X_pca.shape[1]):
            final_km = KMeans(n_clusters=k_final, init=init_centers, n_init=1, max_iter=100, random_state=self.random_state)
        else:
            final_km = KMeans(n_clusters=k_final, init="k-means++", n_init=10, random_state=self.random_state)

        full_labels = final_km.fit_predict(X_pca)
        self.final_kmeans_ = final_km
        self.labels_ = full_labels
        self.cluster_centers_ = final_km.cluster_centers_

        # Invert centroids to original feature units
        if self.pca_ is not None:
            # PCA inverse transform
            scaled_centroids = self.pca_.inverse_transform(self.cluster_centers_)
        else:
            scaled_centroids = self.cluster_centers_

        orig_centroids = self.scaler_.inverse_transform(scaled_centroids)
        # Invert winsorization and skew if needed for real units
        final_orig_centroids = np.zeros_like(orig_centroids)
        for j, col in enumerate(self.feature_names_in_):
            c_vals = orig_centroids[:, j]
            tx = self.skew_transforms_.get(col, "identity")
            if tx == "log1p":
                c_vals = np.expm1(c_vals)
            elif isinstance(tx, PowerTransformer):
                try:
                    c_vals = tx.inverse_transform(c_vals.reshape(-1, 1)).flatten()
                except Exception:
                    pass
            final_orig_centroids[:, j] = c_vals

        self.cluster_centers_original_ = final_orig_centroids

        self.timing_breakdown_["full_refinement_ms"] = round((time.perf_counter() - t_refine0) * 1000.0, 2)
        self.timing_breakdown_["total_runtime_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
        return self

    def predict(self, X_input: Any) -> np.ndarray:
        """Predicts cluster labels for input samples."""
        if self.final_kmeans_ is None:
            raise ValueError("PulseCluster model has not been fitted.")
        
        if isinstance(X_input, pd.DataFrame):
            df_in = X_input.copy()
        else:
            df_in = pd.DataFrame(X_input, columns=[f"feat_{i}" for i in range(X_input.shape[1])])

        X_pca_in = self._transform_new_data(df_in)
        return self.final_kmeans_.predict(X_pca_in)

    def compute_customer_confidence(self, X_pca_samples: np.ndarray) -> np.ndarray:
        """
        Step 8: Per-customer confidence = margin between nearest and second-nearest centroid,
        normalized to 0-100.
        margin = (dist_second - dist_nearest) / (dist_second + 1e-6) * 100
        """
        if self.cluster_centers_ is None:
            raise ValueError("PulseCluster model has not been fitted.")

        centers = self.cluster_centers_  # (k, n_pca)
        confidences = []
        for sample in X_pca_samples:
            dists = np.linalg.norm(centers - sample, axis=1)
            sorted_dists = np.sort(dists)
            d1 = sorted_dists[0]
            d2 = sorted_dists[1] if len(sorted_dists) > 1 else d1 + 1.0
            
            # Margin metric normalized to 0-100
            # If a point is equidistant to two centroids, margin=0 => confidence=0
            # If distance to second centroid is far greater, margin -> 100
            margin = (d2 - d1) / (d2 + 1e-6)
            score = max(0.0, min(100.0, margin * 100.0))
            confidences.append(round(score, 1))

        return np.array(confidences, dtype=np.float32)

    def predict_with_confidence(self, X_input: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (predicted_cluster, confidence_score_0_to_100, distance_to_nearest_centroid).
        """
        if isinstance(X_input, pd.DataFrame):
            df_in = X_input.copy()
        else:
            df_in = pd.DataFrame(X_input, columns=[f"feat_{i}" for i in range(X_input.shape[1])])

        X_pca_in = self._transform_new_data(df_in)
        labels = self.final_kmeans_.predict(X_pca_in)
        confs = self.compute_customer_confidence(X_pca_in)
        
        # Distances to nearest centroid
        nearest_dists = []
        for sample in X_pca_in:
            dists = np.linalg.norm(self.cluster_centers_ - sample, axis=1)
            nearest_dists.append(round(float(np.min(dists)), 3))

        return labels, confs, np.array(nearest_dists, dtype=np.float32)
