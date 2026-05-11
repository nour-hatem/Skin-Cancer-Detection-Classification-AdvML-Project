import sys
import time
import requests
from pathlib import Path

BASE = "http://localhost:8000"
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
VALID_RISKS = {"HIGH", "MEDIUM", "LOW"}


def wait_for_api(retries=10, delay=5):
    print("Waiting for API to be ready...")
    for i in range(retries):
        try:
            r = requests.get(f"{BASE}/health", timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print("[PASS] API is ready\n")
                return
        except Exception:
            pass
        print(f"  attempt {i+1}/{retries} — retrying in {delay}s")
        time.sleep(delay)
    print("[FAIL] API did not become ready in time")
    sys.exit(1)


def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok", f"API degraded: {data}"
    assert data["services"]["classifier"] == "ok", f"Classifier not reachable: {data}"
    print("[PASS] GET /health — api + classifier both ok")


def test_predict(image_path: str):
    path = Path(image_path)
    assert path.exists(), f"Image not found: {image_path}"

    with open(path, "rb") as f:
        r = requests.post(
            f"{BASE}/predict",
            files={"file": (path.name, f, "image/jpeg")},
            timeout=30,
        )

    assert r.status_code == 200, f"Predict failed {r.status_code}: {r.text}"
    d = r.json()

    assert d["lesion"] in CLASS_NAMES,        f"Invalid lesion: {d['lesion']}"
    assert d["risk"] in VALID_RISKS,          f"Invalid risk: {d['risk']}"
    assert 0.0 <= d["confidence"] <= 1.0,     f"Bad confidence: {d['confidence']}"
    assert 0.0 <= d["mel_prob"] <= 1.0,       f"Bad mel_prob: {d['mel_prob']}"
    assert isinstance(d["all_probs"], dict),   "all_probs missing"
    assert len(d["all_probs"]) == 7,          f"Expected 7 probs, got {len(d['all_probs'])}"
    assert "disclaimer" in d and d["disclaimer"], "Disclaimer missing"
    assert "lesion_name" in d,                "lesion_name missing"
    assert "urgency_message" in d,            "urgency_message missing"

    print(f"[PASS] POST /predict")
    print(f"       lesion       : {d['lesion']} ({d['lesion_name']})")
    print(f"       risk         : {d['risk']}")
    print(f"       confidence   : {d['confidence']:.4f}")
    print(f"       mel_prob     : {d['mel_prob']:.4f}")
    print(f"       triggered    : {d['mel_threshold_triggered']}")
    print(f"       all_probs    : {d['all_probs']}")
    print(f"       disclaimer   : {d['disclaimer'][:60]}...")


def test_invalid_file():
    r = requests.post(
        f"{BASE}/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        timeout=10,
    )
    assert r.status_code in (400, 422, 500, 502), \
        f"Expected error on invalid input, got {r.status_code}"
    print("[PASS] Invalid file correctly rejected")


def test_swagger():
    r = requests.get(f"{BASE}/docs")
    assert r.status_code == 200
    print("[PASS] GET /docs — Swagger UI accessible")


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not image_path:
        print("Usage: python test_api.py <path_to_skin_image.jpg>")
        sys.exit(1)

    wait_for_api()
    test_health()
    test_predict(image_path)
    test_invalid_file()
    test_swagger()
    print("\nAll tests passed.")