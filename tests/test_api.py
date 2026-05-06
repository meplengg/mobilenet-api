from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict():
    with open("test.jpg", "rb") as f:
        res = client.post(
            "/predict",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )
    assert res.status_code == 200

def test_invalid():
    res = client.post(
        "/predict",
        files={"file": ("x.txt", b"abc", "text/plain")}
    )
    assert res.status_code == 400