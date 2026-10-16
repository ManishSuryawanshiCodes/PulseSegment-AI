import os
import sys
import pytest
import numpy as np
import pandas as pd
from io import BytesIO
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from models import DatasetConfig
from pipeline import PulseSegmentPipeline
from dataset_manager import DatasetManager, SAFE_ID_REGEX, generate_latent_credit_card_demo
from database import Database

client = TestClient(app)


# =========================================================================
# Item 1: DATA TRUTH
# =========================================================================
def test_item1_data_truth_kaggle_file_balance():
    """Verify Kaggle cc_general.csv has C10001 BALANCE = 40.90, not synthetic 1551.4."""
    cc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cc_general.csv"))
    assert os.path.exists(cc_path), f"cc_general.csv must exist at {cc_path}"

    df = pd.read_csv(cc_path)
    row_c10001 = df[df["CUST_ID"] == "C10001"]
    assert not row_c10001.empty, "Row C10001 must exist in cc_general.csv"
    val = float(row_c10001["BALANCE"].values[0])
    assert abs(val - 40.90) < 0.01, f"Expected real balance 40.90, got {val}"


def test_item1_data_truth_synthetic_separation():
    """Verify is_synthetic flag behavior and synthetic_demo.csv path."""
    dm = DatasetManager(data_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))
    datasets = dm.list_datasets()
    sample_ds = next((d for d in datasets if d.id == "sample_credit_card"), None)
    assert sample_ds is not None

    # Since cc_general.csv exists and has C10001 balance 40.90, is_synthetic must be False
    assert sample_ds.is_synthetic is False

    # Calling generator: writes to synthetic_demo.csv and NOT to cc_general.csv
    synthetic_path = os.path.join(dm.data_dir, "synthetic_demo.csv")
    df_synth = generate_latent_credit_card_demo()
    df_synth.to_csv(synthetic_path, index=False)
    assert os.path.exists(synthetic_path)

    # Re-verify cc_general.csv was NOT overwritten
    cc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cc_general.csv"))
    df = pd.read_csv(cc_path)
    row_c10001 = df[df["CUST_ID"] == "C10001"]
    assert abs(float(row_c10001["BALANCE"].values[0]) - 40.90) < 0.01


# =========================================================================
# Item 2c: Persona Traits |z| >= 0.5 only and fallback note
# =========================================================================
def test_item2c_persona_traits_filtering():
    """
    Persona traits: show only features with |z| >= 0.5, max 3 above and 3 below average.
    Never show features near 0. If fewer than 2 pass, set distinguishing_note.
    """
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "CUST_ID": [f"C_{i}" for i in range(n)],
        "BIG_HIGH": np.concatenate([rng.normal(100, 2, n // 2), rng.normal(10, 2, n // 2)]),
        "BIG_LOW": np.concatenate([rng.normal(5, 1, n // 2), rng.normal(80, 1, n // 2)]),
        "TINY_DIFF_1": rng.normal(50, 0.01, n),
        "TINY_DIFF_2": rng.normal(10, 0.01, n),
        "NEAR_ZERO": rng.normal(0, 0.005, n),
    })

    pipe = PulseSegmentPipeline(df=df, config=DatasetConfig(id_column="CUST_ID", n_clusters=2), dataset_id="test_traits")
    personas_res = pipe.get_growth_personas()
    personas = personas_res["personas"]

    for persona in personas:
        above = persona.top_positive_features
        below = persona.top_negative_features

        assert len(above) <= 3, "Max 3 distinguishing traits above average"
        assert len(below) <= 3, "Max 3 distinguishing traits below average"

        for trait in above:
            assert trait.z_score >= 0.5, f"Trait {trait.feature} has z={trait.z_score} < 0.5"
        for trait in below:
            assert trait.z_score <= -0.5, f"Trait {trait.feature} has z={trait.z_score} > -0.5"

        total_traits = len(above) + len(below)
        if total_traits < 2:
            assert persona.distinguishing_note == "This group differs only slightly from average."


# =========================================================================
# Item 2d: DBSCAN Degenerate Handling
# =========================================================================
def test_item2d_dbscan_degenerate_outcome():
    """
    If DBSCAN finds < 2 clusters or > 60% noise:
    returns metrics null with reason 'Density method found no clear groups'.
    Not eligible for winner.
    """
    rng = np.random.RandomState(99)
    df = pd.DataFrame({f"FEAT_{i}": rng.uniform(0, 10000, 150) for i in range(4)})
    pipe = PulseSegmentPipeline(df=df, config=DatasetConfig(n_clusters=3), dataset_id="test_dbscan_deg")

    # Force degenerate outcome by running with tiny eps
    pipe._estimate_dbscan_eps = lambda X, min_samples=15: 0.0001
    res = pipe.run_experiments()

    dbscan_runs = [e for e in res["experiments"] if e["algorithm"] == "DBSCAN"]
    for run in dbscan_runs:
        if run["cluster_count"] < 2 or run["noise_percentage"] > 60.0:
            assert run["silhouette_score"] is None
            assert run["davies_bouldin_index"] is None
            assert run["calinski_harabasz_score"] is None
            assert run["status"] == "degenerate"
            assert run["reason"] == "Density method found no clear groups"

    # Winner must NEVER be a degenerate DBSCAN run
    winner = res["winning_experiment"]
    assert winner["status"] == "success"
    assert winner["silhouette_score"] is not None


# =========================================================================
# Item 2h: Suggested k (k >= 3, min size >= 3%, composite rank)
# =========================================================================
def test_item2h_suggested_k_selection():
    """
    Suggested k: never auto-pick k=2 just because silhouette is highest there.
    Uses composite (silhouette, CH, DBI rank average) with min cluster size >= 3% and k >= 3.
    """
    rng = np.random.RandomState(42)
    clusters = []
    for center in [10, 50, 100, 200]:
        clusters.append(rng.normal(center, 2, size=(50, 4)))
    X = np.vstack(clusters)
    df = pd.DataFrame(X, columns=[f"F_{i}" for i in range(4)])

    pipe = PulseSegmentPipeline(df=df, config=DatasetConfig(n_clusters=4), dataset_id="test_k_sweep")
    k_sweep, suggested_k, reason = pipe.run_k_sweep()

    assert suggested_k >= 3, f"Suggested k must be >= 3, got {suggested_k}"
    assert "selected by composite rank" in reason
    assert "minimum threshold" in reason or "balanced" in reason
    assert suggested_k != 2


# =========================================================================
# Item 3: METHOD VALIDITY - Common-Space Clarity
# =========================================================================
def test_item3_common_space_clarity_computation():
    """
    For every run, compute common-space clarity (silhouette, DBI, CHI on DS1_Original scaled).
    Winner is selected using common-space metrics.
    """
    rng = np.random.RandomState(42)
    data = {f"FEAT_{i}": rng.normal(0, 1, 100) for i in range(5)}
    df = pd.DataFrame(data)

    pipe = PulseSegmentPipeline(df=df, config=DatasetConfig(n_clusters=3), dataset_id="test_common_space")
    res = pipe.run_experiments()

    for exp in res["experiments"]:
        if exp["status"] == "success":
            assert exp["common_space_silhouette"] is not None
            assert exp["common_space_dbi"] is not None
            assert exp["common_space_chi"] is not None

    winner = res["winning_experiment"]
    assert winner["common_space_silhouette"] is not None
    assert winner["composite_score"] is not None


# =========================================================================
# Item 5: SECURITY
# =========================================================================
def test_item5_security_oversized_file():
    """Uploads with content exceeding 50MB must be rejected with 422."""
    from fastapi import UploadFile
    from starlette.datastructures import Headers

    class LargeStream(BytesIO):
        def __init__(self, size_bytes):
            super().__init__()
            self.total = size_bytes
            self.read_so_far = 0

        def read(self, size=-1):
            if self.read_so_far >= self.total:
                return b""
            n = min(size if size > 0 else 65536, self.total - self.read_so_far)
            self.read_so_far += n
            return b"A" * n

    oversized_stream = LargeStream(52 * 1024 * 1024)
    upload_file = UploadFile(file=oversized_stream, filename="too_big.csv", headers=Headers())
    dm = DatasetManager(data_dir="backend/data")

    with pytest.raises(Exception) as exc_info:
        dm.upload_dataset(upload_file)
    assert "exceeds maximum allowed limit" in str(exc_info.value)


def test_item5_security_fake_csv():
    """Uploading a non-CSV binary file disguised as .csv must be rejected."""
    from fastapi import UploadFile
    from starlette.datastructures import Headers

    fake_binary_content = b"\x7f\x45\x4c\x46\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00"  # Binary header
    upload_file = UploadFile(file=BytesIO(fake_binary_content), filename="invalid_binary.csv", headers=Headers())
    dm = DatasetManager(data_dir="backend/data")

    with pytest.raises(Exception) as exc_info:
        dm.upload_dataset(upload_file)
    assert "Invalid CSV file content" in str(exc_info.value) or "Failed to parse" in str(exc_info.value)


def test_item5_security_formula_injection():
    """CSV export must neutralise formula injection by prefixing =, +, -, @ with apostrophe."""
    df_vuln = pd.DataFrame({
        "CUST_ID": ["1", "2", "3", "4"],
        "VAL": [10.0, 20.0, 30.0, 40.0],
        "PAYLOAD": ["=cmd|' /C calc'!A0", "+12345", "-@sum(1,1)", "@SUM(1,2)"],
    })
    pipe = PulseSegmentPipeline(df=df_vuln, config=DatasetConfig(id_column="CUST_ID", n_clusters=2), dataset_id="test_formula")
    pipe.run_experiments()

    csv_output = pipe.export_segments_csv()
    for line in csv_output.splitlines():
        if "=cmd" in line:
            assert "'=cmd" in line
        if "+12345" in line:
            assert "'+12345" in line
        if "-@sum" in line:
            assert "'-@sum" in line
        if "@SUM" in line:
            assert "'@SUM" in line


def test_item5_security_bad_id_path_traversal():
    """Dataset and job IDs must be validated against safe regex; path traversal rejected."""
    assert not SAFE_ID_REGEX.match("../../etc/passwd")
    assert not SAFE_ID_REGEX.match("..\\windows\\system32")
    assert not SAFE_ID_REGEX.match("test/nested")
    assert not SAFE_ID_REGEX.match("bad;id")
    assert SAFE_ID_REGEX.match("sample_credit_card")
    assert SAFE_ID_REGEX.match("123e4567-e89b-12d3-a456-426614174000")

    # API test for bad dataset ID
    resp = client.get("/api/datasets/..%2f..%2fetc%2fpasswd")
    assert resp.status_code in [400, 404]

    resp_json = client.get("/api/datasets/invalid;id/model.json")
    assert resp_json.status_code == 400


def test_item5_security_model_json_and_headers():
    """Test model.json export endpoint and model.joblib warning header."""
    # Model JSON endpoint
    resp = client.get("/api/datasets/sample_credit_card/model.json")
    assert resp.status_code == 200
    data = resp.json()
    assert "centroids" in data
    assert "feature_names" in data
    assert "scaler" in data
    assert "mean" in data["scaler"]

    # Model joblib download security warning header
    resp_joblib = client.get("/api/datasets/sample_credit_card/model.joblib")
    assert resp_joblib.status_code == 200
    assert "X-Security-Warning" in resp_joblib.headers
    assert "trusted sources" in resp_joblib.headers["X-Security-Warning"]


def test_item5_security_delete_dataset():
    """DELETE /api/datasets/{id} deletes dataset and cleans up cache."""
    # Upload a valid CSV dataset with at least 3 numeric columns
    csv_data = "ID,FEAT1,FEAT2,FEAT3\n" + "\n".join([f"ID_{i},{i*2},{i*3},{i*4}" for i in range(120)])
    upload_resp = client.post(
        "/api/datasets/upload",
        files={"file": ("test_delete.csv", csv_data.encode("utf-8"), "text/csv")},
    )
    assert upload_resp.status_code == 200
    d_id = upload_resp.json()["id"]

    # Call DELETE endpoint
    del_resp = client.delete(f"/api/datasets/{d_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # Verify 404 after deletion on dataset profile
    get_resp = client.get(f"/api/datasets/{d_id}/profile")
    assert get_resp.status_code == 404
