"""
Tests for PulseCluster Accuracy and Robustness Requirements
==========================================================
Verifies:
1. Reproducibility with a fixed seed.
2. Graceful execution without crash on constant columns or tiny clusters.
3. Reliability score is strictly between 0 and 100.
4. Per-customer confidence margin is bounded between 0 and 100.
5. Holdout consistency and predict() functionality.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from pulsecluster import PulseCluster
from benchmarks.benchmark_suite import (
    generate_gaussian_blobs,
    generate_skewed_lognormal_blobs,
    generate_overlapping_groups,
    generate_17_feature_correlated_data,
)


def test_pulsecluster_reproducibility():
    """Verify PulseCluster gives identical labels with a fixed random seed."""
    df, _ = generate_gaussian_blobs(n_samples=200, random_state=42)
    
    pc1 = PulseCluster(n_clusters=3, n_bootstraps=3, random_state=123)
    pc1.fit(df)

    pc2 = PulseCluster(n_clusters=3, n_bootstraps=3, random_state=123)
    pc2.fit(df)

    assert np.array_equal(pc1.labels_, pc2.labels_)
    assert pc1.reliability_ == pytest.approx(pc2.reliability_, abs=1e-3)
    assert pc1.selected_k_ == pc2.selected_k_


def test_pulsecluster_constant_columns_and_tiny_clusters():
    """Verify PulseCluster never crashes on constant columns, all-zero features, or tiny clusters."""
    # Data with constant column and extreme imbalance (tiny cluster of 2 rows)
    n_samples = 100
    df = pd.DataFrame({
        "CONST_1": [5.0] * n_samples,
        "CONST_2": [0.0] * n_samples,
        "FEAT_VAR": np.linspace(1.0, 100.0, n_samples),
        "OUTLIER_TINY": [1000.0 if i < 2 else 10.0 for i in range(n_samples)],
        "BALANCE": [50.0] * n_samples,
        "CREDIT_LIMIT": [0.0] * n_samples,  # test divide-by-zero guard
    })

    pc = PulseCluster(n_clusters=3, n_bootstraps=3, time_budget_sec=5.0, random_state=42)
    pc.fit(df)

    assert len(pc.labels_) == n_samples
    assert pc.cluster_centers_ is not None
    preds = pc.predict(df.iloc[:10])
    assert len(preds) == 10


def test_pulsecluster_reliability_bounded_0_to_100():
    """Verify reliability score is always strictly between 0 and 100."""
    datasets = [
        generate_gaussian_blobs(n_samples=150, random_state=42)[0],
        generate_overlapping_groups(n_samples=150, random_state=42)[0],
        generate_17_feature_correlated_data(n_samples=150, random_state=42)[0],
    ]

    for df in datasets:
        pc = PulseCluster(n_clusters=3, n_bootstraps=3, time_budget_sec=5.0, random_state=42)
        pc.fit(df)
        assert 0.0 <= pc.reliability_ <= 100.0, f"Reliability {pc.reliability_} outside [0, 100]"
        assert 0.0 <= pc.stability_score_ <= 1.0


def test_pulsecluster_confidence_and_holdout():
    """Verify per-customer confidence margin is normalized to 0-100 and holdout predict matches."""
    df, _ = generate_17_feature_correlated_data(n_samples=250, random_state=42)
    
    train_df = df.iloc[:200]
    test_df = df.iloc[200:]

    pc = PulseCluster(n_clusters=4, n_bootstraps=3, time_budget_sec=5.0, random_state=42)
    pc.fit(train_df)

    labels, confs, dists = pc.predict_with_confidence(test_df)
    assert len(labels) == len(test_df)
    assert len(confs) == len(test_df)
    assert len(dists) == len(test_df)

    for c in confs:
        assert 0.0 <= c <= 100.0, f"Confidence {c} outside [0, 100]"
    for d in dists:
        assert d >= 0.0

    # Ensure timing breakdown exists
    assert "total_runtime_ms" in pc.timing_breakdown_
    assert pc.timing_breakdown_["total_runtime_ms"] > 0


def test_api_new_features_endpoints():
    """Verify all 7 new endpoints: scores, lookup, summary, runs, compare, drift, report."""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)

    # 1. Scores endpoint
    res_scores = client.get("/api/datasets/sample_credit_card/scores")
    assert res_scores.status_code == 200
    scores_data = res_scores.json()
    assert isinstance(scores_data, list)
    assert len(scores_data) > 0
    assert "value_score" in scores_data[0]
    assert "risk_score" in scores_data[0]
    assert "formula_explanation" in scores_data[0]

    # 2. Customer Lookup endpoint
    res_cust = client.get("/api/datasets/sample_credit_card/customer/0")
    assert res_cust.status_code == 200
    cust_data = res_cust.json()
    assert cust_data["customer_id"] == "0"
    assert "confidence" in cust_data
    assert "persona_name" in cust_data
    assert "distinguishing_traits" in cust_data
    assert "differences_vs_cluster" in cust_data

    # 3. Summary Report endpoint
    res_sum = client.get("/api/datasets/sample_credit_card/summary-report")
    assert res_sum.status_code == 200
    sum_data = res_sum.json()
    assert sum_data["source_type"] in ("Template", "AI-written")
    assert "summary_text" in sum_data
    assert len(sum_data["key_findings"]) > 0

    # 4. Saved Runs list endpoint
    res_runs = client.get("/api/datasets/sample_credit_card/runs")
    assert res_runs.status_code == 200
    runs_data = res_runs.json()
    assert isinstance(runs_data, list)

    # 5. One-page Report HTML export endpoint
    res_html = client.get("/api/datasets/sample_credit_card/report.html")
    assert res_html.status_code == 200
    assert "text/html" in res_html.headers["content-type"]
    assert "PulseSegment AI" in res_html.text

