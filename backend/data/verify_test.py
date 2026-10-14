import urllib.request
import json

def verify_rfm(ds_id):
    url = f"http://127.0.0.1:8000/api/datasets/{ds_id}/personas"
    req = urllib.request.urlopen(url)
    data = json.loads(req.read().decode())
    print("Winning model:", data.get("winning_model"))
    print("Domain template:", data.get("domain_template"))
    print("Silhouette:", data.get("winning_silhouette"))
    print("DBI:", data.get("winning_dbi"))
    print("-" * 60)
    for p in data["personas"]:
        print(f"Cluster {p['id']}: {p['name']}")
        print(f"  Count: {p['count']} ({p['percentage']}%)")
        traits = [f"{t['feature']} (z={t['z_score']:+.2f}, mean={t['cluster_mean']:.1f} vs {t['global_mean']:.1f})" for t in p.get("top_positive_features", []) + p.get("top_negative_features", [])]
        print(f"  Top traits: {', '.join(traits)}")
        campaigns = [c["title"] for c in p.get("nba_campaigns", [])]
        print(f"  NBA Campaigns: {', '.join(campaigns)}")
        print()

if __name__ == "__main__":
    print("=== UPLOADED DATASET (ecommerce_rfm_test.csv) ===")
    verify_rfm("ds_b5f0ad5f")
    print("\n=== SAMPLE DATASET (sample_credit_card) ===")
    verify_rfm("sample_credit_card")


