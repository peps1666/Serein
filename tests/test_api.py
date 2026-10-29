import asyncio

from fastapi.testclient import TestClient
import httpx

from serein.app import create_app
from serein.config import Settings
from serein.provider import OpenAIProvider


def test_api_auth_confirmations_and_delete(tmp_path):
    settings = Settings(db_path=tmp_path / "api.sqlite3", users={"local": "local-password-123"})
    app = create_app(settings, with_ui=False)
    auth = ("local", "local-password-123")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/api/chat", json={"text": "private"}).status_code == 401
        assert client.post("/api/chat", auth=("unknown", "bad"), json={"text": "private"}).status_code == 401
        assert app.state.core.store.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
        assert client.put("/api/preferences", auth=auth, json={"nickname": "小林"}).status_code == 422
        assert client.put("/api/preferences", auth=auth, json={"nickname": "小林", "confirmed": False}).status_code == 422
        assert client.put("/api/preferences", auth=auth, json={"nickname": "小林", "confirmed": True}).status_code == 200
        result = client.post("/api/chat", auth=auth, json={"text": "今天被领导批评了"})
        assert result.status_code == 200 and result.json()["status"] == "completed"
        assert result.headers["cache-control"] == "no-store"
        assert client.get("/api/state", auth=auth).json()["preferences"]["content"]["nickname"] == "小林"
        assert client.post("/api/delete-all", auth=auth, json={"confirmed": False}).status_code == 422
        assert client.post("/api/delete-all", auth=auth, json={"confirmed": True}).status_code == 200
        data = client.get("/api/state", auth=auth).json()
        assert data["history"] == [] and data["preferences"] is None
        assert client.put("/api/mode", auth=auth, json={"no_save": True}).status_code == 200
        assert client.put("/api/preferences", auth=auth, json={"confirmed": True}).status_code == 409


def test_tools_followup_and_inbox_are_authenticated_and_deleted(tmp_path):
    settings = Settings(db_path=tmp_path / 'api.sqlite3', users={'alice': 'alice-password-123', 'bob': 'bob-password-123'})
    app = create_app(settings, with_ui=False)
    alice, bob = ('alice', 'alice-password-123'), ('bob', 'bob-password-123')
    with TestClient(app) as client:
        assert client.post('/api/tools/create_note', json={'arguments': {'title': 'A', 'content': 'B'}}).status_code == 401
        saved = client.post('/api/tools/create_note', auth=alice, json={'arguments': {'title': 'A', 'content': 'B'}}).json()
        assert saved['ok']
        assert not client.post('/api/tools/get_note', auth=bob, json={'arguments': {'id': saved['id']}}).json()['ok']
        reminder = client.post('/api/tools/create_reminder', auth=alice,
                               json={'arguments': {'content': '测试提醒', 'due': '2099-01-01 09:00'}}).json()
        assert reminder['ok']
        assert client.put('/api/followup', auth=alice, json={'enabled': True, 'hour': 20, 'minute': 0}).json()['enabled']
        assert client.put('/api/mode', auth=alice, json={'no_save': True}).status_code == 200
        assert not client.post('/api/tools/create_note', auth=alice, json={'arguments': {'title': '不保存', 'content': '秘密'}}).json()['ok']
        assert client.get('/api/notifications', auth=alice).json() == {'notifications': []}
        assert client.put('/api/followup', auth=alice, json={'enabled': False}).status_code == 200
        assert client.post('/api/delete-all', auth=alice, json={'confirmed': True}).status_code == 200
        assert app.state.core.store.notes('alice') == [] and app.state.core.store.reminders('alice') == []


def test_cloud_requires_explicit_configuration():
    try:
        Settings(provider="openai", api_key="fake-test-key", model="test-model")
    except ValueError as exc:
        assert "CLOUD_ACKNOWLEDGED" in str(exc)
    else:
        raise AssertionError("Cloud mode must not start without acknowledgement")


def test_official_sdk_request_with_mock_transport():
    import json
    from openai import AsyncOpenAI

    async def scenario():
        seen = []

        def handle(request):
            body = json.loads(request.content)
            seen.append(body)
            assert body["store"] is False
            assert body["model"] == "test-model"
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1,
                "model": "test-model", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "我听着。"}}]})

        settings = Settings(provider="openai", api_key="test-key", model="test-model", cloud_acknowledged=True)
        provider = OpenAIProvider(settings)
        await provider.client.close()
        provider.client = AsyncOpenAI(api_key="test-key", base_url="https://example.invalid/v1", max_retries=0,
                                      http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        result = await provider.generate([{"role": "user", "content": "合成场景"}])
        assert result.text == "我听着。"
        assert result.model == "test-model"
        assert len(seen) == 1
        await provider.close()

    asyncio.run(scenario())
