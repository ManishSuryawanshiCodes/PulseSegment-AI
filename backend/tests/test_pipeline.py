import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import DatasetConfig
from pipeline import PulseSegmentPipeline


def create_mock_dataset(n_samples=200, n_features=6, random_state=42, add_clusters=True):
    """Generates synthetic customer behavioral data with configurable dimensions and clusters."""
    rng = np.random.RandomState(random_state)
    data = {}
    data["CUSTOMER_ID"] = [f"CUST_{i+1000}" for i in range(n_samples)]

    if add_clusters:
        # Create 3 separable clusters with distinct per-feature variance
        cluster_assignment = rng.choice([0, 1, 2], size=n_samples, p=[0.4, 0.35, 0.25])
        for j in range(n_features):
            col_name = f"FEATURE_{j+1}"
            # Vary centers per feature to avoid artificial all-feature multicollinearity
            c0 = rng.uniform(10, 50)
            c1 = rng.uniform(100, 300)
            c2 = rng.uniform(500, 1000)
            centers = [c0, c1, c2]
            col_vals = [centers[c] + rng.normal(0, 8) for c in cluster_assignment]
            data[col_name] = col_vals
    else:
        for j in range(n_features):
            data[f"FEATURE_{j+1}"] = rng.uniform(10, 100, size=n_samples)

    return pd.DataFrame(data)


# -------------------------------------------------------------------------
# Test (a): Same dataset gives identical results with a fixed random_state
# -------------------------------------------------------------------------
def test_reproducibility_same_dataset():
    df1 = create_mock_dataset(n_samples=150, n_features=6, random_state=42)
    df2 = df1.copy()

    cfg1 = DatasetConfig(n_clusters=3, iqr_multiplier=1.5, corr_threshold=0.85)
    cfg2 = DatasetConfig(n_clusters=3, iqr_multiplier=1.5, corr_threshold=0.85)

    pipe1 = PulseSegmentPipeline(df=df1, config=cfg1, dataset_id="test_repro_1")
    pipe2 = PulseSegmentPipeline(df=df2, config=cfg2, dataset_id="test_repro_2")

    res1 = pipe1.run_experiments()
    res2 = pipe2.run_experiments()

    assert len(res1["experiments"]) in (16, 17)
    assert len(res2["experiments"]) in (16, 17)

    # Compare winning model metrics bit-for-bit
    w1 = res1["winning_experiment"]
    w2 = res2["winning_experiment"]
    assert w1["algorithm"] == w2["algorithm"]
    assert w1["dataset_name"] == w2["dataset_name"]
    assert w1["silhouette_score"] == pytest.approx(w2["silhouette_score"], abs=1e-5)
    assert w1["davies_bouldin_index"] == pytest.approx(w2["davies_bouldin_index"], abs=1e-5)
    assert w1["calinski_harabasz_score"] == pytest.approx(w2["calinski_harabasz_score"], abs=1e-2)
    assert np.array_equal(pipe1._winning_labels, pipe2._winning_labels)

    # Compare all 16 experiment silhouette scores
    for exp1, exp2 in zip(res1["experiments"], res2["experiments"]):
        assert exp1["algorithm"] == exp2["algorithm"]
        assert exp1["dataset_name"] == exp2["dataset_name"]
        if exp1["silhouette_score"] is not None:
            assert exp1["silhouette_score"] == pytest.approx(exp2["silhouette_score"], abs=1e-5)
        else:
            assert exp2["silhouette_score"] is None


# -------------------------------------------------------------------------
# Test (b): Two different datasets give different metrics
# -------------------------------------------------------------------------
def test_different_datasets_give_different_metrics():
    df_clustered = create_mock_dataset(n_samples=200, n_features=6, random_state=101, add_clusters=True)
    df_uniform = create_mock_dataset(n_samples=200, n_features=6, random_state=999, add_clusters=False)

    pipe_clustered = PulseSegmentPipeline(df=df_clustered, config=DatasetConfig(n_clusters=3), dataset_id="clustered")
    pipe_uniform = PulseSegmentPipeline(df=df_uniform, config=DatasetConfig(n_clusters=3), dataset_id="uniform")

    res_clustered = pipe_clustered.run_experiments()
    res_uniform = pipe_uniform.run_experiments()

    sil_clustered = res_clustered["winning_experiment"]["silhouette_score"]
    sil_uniform = res_uniform["winning_experiment"]["silhouette_score"]

    assert sil_clustered is not None
    assert sil_uniform is not None
    # Clustered dataset must have a noticeably different (typically higher) silhouette than uniform noise
    assert abs(sil_clustered - sil_uniform) > 0.05
    assert res_clustered["winning_experiment"]["davies_bouldin_index"] != res_uniform["winning_experiment"]["davies_bouldin_index"]


# -------------------------------------------------------------------------
# Test (c): CSV with missing values and a text column is handled properly
# -------------------------------------------------------------------------
def test_missing_values_and_text_column_handling():
    df = create_mock_dataset(n_samples=180, n_features=5, random_state=77)

    # Add text categorical column
    regions = ["North America", "EMEA", "APAC", "LATAM"]
    df["REGION"] = [regions[i % len(regions)] for i in range(len(df))]

    # Inject missing values (NaN) in numerical features
    df.loc[5:15, "FEATURE_1"] = np.nan
    df.loc[20:25, "FEATURE_3"] = np.nan

    initial_missing_count = int(df[["FEATURE_1", "FEATURE_3"]].isna().sum().sum())
    assert initial_missing_count > 0

    cfg = DatasetConfig(
        id_column="CUSTOMER_ID",
        imputation_k=5,
        iqr_multiplier=1.5,
        n_clusters=3,
        domain_template="generic",
    )

    pipe = PulseSegmentPipeline(df=df, config=cfg, dataset_id="test_missing_handling")
    pipe.preprocess()

    # Preprocessed dataframe must not contain any NaNs
    processed_df = pipe._processed_df
    assert processed_df is not None
    assert processed_df.isna().sum().sum() == 0

    # Verify ID column and categorical text column were excluded from numerical normalization
    assert "CUSTOMER_ID" not in processed_df.columns
    assert "REGION" not in processed_df.columns
    assert processed_df.shape[1] == 5

    # Check summary report logged missing cells accurately
    summary = pipe.get_dataset_summary()
    assert sum(summary["missing_values"].values()) == initial_missing_count
    assert any("KNN Imputation" in step for step in summary["preprocessing_steps"])


# -------------------------------------------------------------------------
# Test (d): Degenerate case (DBSCAN all noise) returns nulls, not a crash
# -------------------------------------------------------------------------
def test_dbscan_all_noise_returns_nulls():
    # Widely dispersed uniform noise points with tiny eps auto-estimation guarantee all noise
    rng = np.random.RandomState(1234)
    df = pd.DataFrame({
        f"FEAT_{i}": rng.uniform(-10000, 10000, size=100) for i in range(4)
    })

    pipe = PulseSegmentPipeline(df=df, config=DatasetConfig(n_clusters=3), dataset_id="degenerate_dbscan")

    # Override DBSCAN estimator to force zero neighborhood capture
    pipe._estimate_dbscan_eps = lambda X, min_samples=15: 0.000001

    res = pipe.run_experiments()
    assert "experiments" in res

    dbscan_exps = [e for e in res["experiments"] if e["algorithm"] == "DBSCAN"]
    assert len(dbscan_exps) == 4

    for exp in dbscan_exps:
        # All noise points mean cluster_count < 2
        if exp["cluster_count"] < 2 or exp["noise_percentage"] >= 99.0:
            assert exp["silhouette_score"] is None
            assert exp["davies_bouldin_index"] is None
            assert exp["calinski_harabasz_score"] is None
            assert exp["status"] == "degenerate"
            assert "noise" in exp["reason"].lower() or "cluster" in exp["reason"].lower() or "density method found no clear groups" in exp["reason"].lower()


# -------------------------------------------------------------------------
# Test (e): Phase 4 feature counts are computed dynamically, not constants
# -------------------------------------------------------------------------
def test_phase4_feature_counts_are_dynamically_computed():
    # Case 1: Dataset with 5 independent features + 1 collinear feature
    rng = np.random.RandomState(42)
    data_5 = {f"FEAT_{i+1}": rng.normal(i * 10, 5, size=120) for i in range(5)}
    df_5 = pd.DataFrame(data_5)
    # Inject deliberate collinear feature with FEAT_1 (|r| ~ 1.0)
    df_5["FEAT_COLLINEAR"] = df_5["FEAT_1"] * 3.0 + 0.5

    pipe_5 = PulseSegmentPipeline(df=df_5, config=DatasetConfig(corr_threshold=0.85))
    fe_5 = pipe_5.feature_engineering()

    # DS1 Original count must match exact included features (6)
    assert len(fe_5["original_features"]) == 6

    # DS2 Correlation count must prune the collinear feature and keep 5
    assert len(fe_5["ds2_correlation_features"]) == 5
    assert len(fe_5["ds2_pruned_details"]) == 1
    for pruned in fe_5["ds2_pruned_details"]:
        assert abs(pruned["correlation"]) >= 0.85
        assert len(pruned["correlated_with"]) > 0

    # DS3 SelectKBest must be min(8, n_features - 1) = min(8, 6 - 1) = 5 (NOT 8!)
    assert len(fe_5["ds3_mutual_info_features"]) == 5

    # DS4 PCA components must be dynamically determined
    assert fe_5["ds4_pca_n_components"] <= 6
    assert len(fe_5["ds4_pca_explained_variance_ratio"]) == fe_5["ds4_pca_n_components"]

    # Case 2: Dataset with 12 features
    data_12 = {f"FEAT_{i+1}": rng.normal(i * 10, 5, size=150) for i in range(12)}
    df_12 = pd.DataFrame(data_12)
    pipe_12 = PulseSegmentPipeline(df=df_12, config=DatasetConfig(corr_threshold=0.85))
    fe_12 = pipe_12.feature_engineering()

    assert len(fe_12["original_features"]) == 12
    # For 12 features, min(8, 12 - 1) = 8
    assert len(fe_5["ds3_mutual_info_features"]) != len(fe_12["ds3_mutual_info_features"])
    assert len(fe_12["ds3_mutual_info_features"]) == 8


# -------------------------------------------------------------------------
# Test (f): RFM fixture with lapsed cluster must be labelled "At-Risk"
# -------------------------------------------------------------------------
def test_rfm_lapsed_cluster_labelled_at_risk():
    """
    Verifies that high RECENCY is inverted in archetype scoring, so that
    high RECENCY + low FREQUENCY maps to 'At-Risk / Lapsed'.
    """
    rng = np.random.RandomState(42)
    n_per_cluster = 60

    # Cluster 0: Champions (low recency, high frequency, high monetary)
    c0 = {
        "CUSTOMER_ID": [f"CHAMP_{i}" for i in range(n_per_cluster)],
        "RECENCY": rng.uniform(2, 10, size=n_per_cluster),
        "FREQUENCY": rng.uniform(20, 40, size=n_per_cluster),
        "MONETARY": rng.uniform(1500, 3000, size=n_per_cluster),
    }

    # Cluster 1: Recent / Promising (low recency, moderate frequency, moderate spend)
    c1 = {
        "CUSTOMER_ID": [f"PROM_{i}" for i in range(n_per_cluster)],
        "RECENCY": rng.uniform(3, 14, size=n_per_cluster),
        "FREQUENCY": rng.uniform(2, 5, size=n_per_cluster),
        "MONETARY": rng.uniform(100, 250, size=n_per_cluster),
    }

    # Cluster 2: Clearly Lapsed / At-Risk (high recency 250-400 days, low frequency 1 order)
    c2 = {
        "CUSTOMER_ID": [f"LAPSED_{i}" for i in range(n_per_cluster)],
        "RECENCY": rng.uniform(250, 400, size=n_per_cluster),
        "FREQUENCY": rng.uniform(1, 2, size=n_per_cluster),
        "MONETARY": rng.uniform(20, 60, size=n_per_cluster),
    }

    df_rfm = pd.concat([pd.DataFrame(c0), pd.DataFrame(c1), pd.DataFrame(c2)], ignore_index=True)

    cfg = DatasetConfig(
        id_column="CUSTOMER_ID",
        domain_template="ecommerce_rfm",
        n_clusters=3,
        iqr_multiplier=2.5,
    )

    pipe = PulseSegmentPipeline(df=df_rfm, config=cfg, dataset_id="test_rfm_lapsed")
    personas_res = pipe.get_growth_personas()

    personas = personas_res["personas"]
    assert len(personas) == 3

    persona_names = [p.name for p in personas]
    assert any("At-Risk" in name for name in persona_names), f"Expected 'At-Risk' in {persona_names}"

    # Verify trait wording in direction map: recency is lower_is_better
    assert personas_res.get("direction_map", {}).get("RECENCY") == "lower_is_better"

    # Find the lapsed persona and check characteristics or distinguishing trait
    at_risk_persona = next(p for p in personas if "At-Risk" in p.name)
    assert abs(at_risk_persona.count - n_per_cluster) <= 3
    assert any("lapsed" in c.lower() or "recency" in c.lower() for c in at_risk_persona.characteristics)


# -------------------------------------------------------------------------
# Test (g): Persona count always equals k for k=3, 4, 5 with unique names
# -------------------------------------------------------------------------
@pytest.mark.parametrize("k", [3, 4, 5])
def test_persona_count_equals_k_for_3_4_5(k):
    """Guarantees number of personas strictly equals n_clusters with unique names."""
    df = create_mock_dataset(n_samples=180, n_features=6, random_state=42)
    cfg = DatasetConfig(
        id_column="CUSTOMER_ID",
        domain_template="ecommerce_rfm",
        n_clusters=k,
    )
    pipe = PulseSegmentPipeline(df=df, config=cfg, dataset_id=f"test_k_{k}")
    res = pipe.get_growth_personas()

    personas = res["personas"]
    assert len(personas) == k, f"Expected {k} personas for k={k}, got {len(personas)}"

    names = [p.name for p in personas]
    assert len(set(names)) == k, f"Names must all be unique, got {names}"

    # Verify box plots are generated for top features
    assert "box_plots" in res
    assert len(res["box_plots"]) > 0


# -------------------------------------------------------------------------
# Test (h): Silhouette constant sample size (5000 applies when N > 8000)
# -------------------------------------------------------------------------
def test_silhouette_sample_constant_definition():
    from pipeline import SILHOUETTE_SAMPLE_THRESHOLD, SILHOUETTE_SAMPLE_SIZE
    assert SILHOUETTE_SAMPLE_THRESHOLD == 8000
    assert SILHOUETTE_SAMPLE_SIZE == 5000

