from fastapi.testclient import TestClient

from serein.app import create_app
from serein.config import Settings


def test_gradio_login_and_chat_callback(tmp_path):
    settings = Settings(db_path=tmp_path / "ui.sqlite3", users={"local": "local-password-123"})
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/chat/config").status_code == 401
        assert client.post("/chat/login", data={"username": "local", "password": "wrong"}).status_code == 400
        assert client.post("/chat/login", data={"username": "local", "password": "local-password-123"}).status_code == 200
        config = client.get("/chat/config")
        assert config.status_code == 200
        assert config.json()["analytics_enabled"] is False
        result = client.post("/chat/gradio_api/run/send", json={"data": ["今天被领导批评了"], "session_hash": "test-local"})
        assert result.status_code == 200, result.text
        data = result.json()["data"]
        assert data[0] == ""
        assert data[1][-1]["role"] == "assistant"
        assert "已回应" in data[3]
