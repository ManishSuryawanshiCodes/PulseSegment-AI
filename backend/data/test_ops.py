import urllib.request
import json

def test_inference_and_exports(ds_id):
    print(f"\n================ TESTING OPERATIONAL ENDPOINTS FOR {ds_id} ================")
    
    # 1. Single Customer Inference
    predict_url = f"http://127.0.0.1:8000/api/datasets/{ds_id}/predict-segment"
    payload = {"features": {"PURCHASES": 4500.0, "ONEOFF_PURCHASES": 3200.0, "MONETARY": 3500.0, "FREQUENCY": 12.0, "RECENCY": 2.0}}
    req = urllib.request.Request(predict_url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        pred_data = json.loads(resp.read().decode())
        print(">> Predict Segment Response:")
        print(f"   Assigned Cluster: {pred_data['cluster_label']}")
        print(f"   Persona Name: {pred_data['persona_name']}")
        print(f"   Confidence: {pred_data['confidence'] * 100:.1f}%")
        print(f"   Recommended NBA: {pred_data['recommended_nba']['title']}")
        print(f"   Top Traits: {pred_data['top_traits']}")

    # 2. Export Segments CSV
    csv_url = f"http://127.0.0.1:8000/api/datasets/{ds_id}/export/segments.csv"
    with urllib.request.urlopen(csv_url) as resp:
        csv_head = resp.read(300).decode("utf-8", errors="ignore")
        print("\n>> Export Segments CSV (first 300 bytes):")
        print(csv_head.strip())

    # 3. Download Trained Model Joblib
    joblib_url = f"http://127.0.0.1:8000/api/datasets/{ds_id}/model.joblib"
    with urllib.request.urlopen(joblib_url) as resp:
        joblib_bytes = resp.read()
        print(f"\n>> Export Model Joblib Bundle:")
        print(f"   Downloaded {len(joblib_bytes)} bytes binary artifact successfully.")

if __name__ == "__main__":
    test_inference_and_exports("ds_b5f0ad5f")
    test_inference_and_exports("sample_credit_card")
