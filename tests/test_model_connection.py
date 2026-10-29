import asyncio
import json
import os

import httpx
from openai import AsyncOpenAI
import pytest

from serein.config import Settings
from serein.models import ChatInput
from serein.provider import Generation, OpenAIProvider, ProviderFailure
from test_core import make_core, run


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in list(os.environ):
        if name.startswith(("SEREIN_", "OPENAI_", "DEEPSEEK_")):
            monkeypatch.delenv(name)


def test_auto_selects_available_deepseek_without_exposing_key(isolated_env, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-test-secret")
    monkeypatch.setenv("SEREIN_CLOUD_ACKNOWLEDGED", "true")
    settings = Settings.from_env()
    assert settings.provider == "deepseek"
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.model == "deepseek-flash"
    assert settings.api_key == "synthetic-test-secret"
    assert "synthetic-test-secret" not in repr(settings)


def test_no_key_requires_configuration_instead_of_fake_reply(isolated_env):
    with pytest.raises(ValueError, match="No model configured"):
        Settings.from_env()
    assert Settings.from_env(demo=True).provider == "demo"


def test_explicit_openai_is_not_silently_changed_to_deepseek(isolated_env, monkeypatch):
    monkeypatch.setenv("SEREIN_PROVIDER", "openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-test-secret")
    monkeypatch.setenv("SEREIN_CLOUD_ACKNOWLEDGED", "true")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_deepseek_compatible_schema_and_token_usage():
    async def scenario():
        def handle(request):
            data = json.loads(request.content)
            assert "store" not in data
            assert data["thinking"] == {"type": "disabled"}
            assert data["max_tokens"] == 400
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1,
                "model": "deepseek-flash", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "这会让人觉得很委屈。"}}],
                "usage": {"prompt_tokens": 123, "completion_tokens": 20, "total_tokens": 143}})
        provider = OpenAIProvider(Settings(provider="deepseek", model="deepseek-flash", api_key="test-key", cloud_acknowledged=True))
        await provider.client.close()
        provider.client = AsyncOpenAI(api_key="test-key", base_url="https://example.invalid", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        result = await provider.generate([{"role": "user", "content": "合成输入"}])
        assert result.input_tokens == 123 and result.output_tokens == 20
        assert result.text == "这会让人觉得很委屈。"
        await provider.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("status, expected", [(401, "model_auth_error"), (402, "model_balance_error"),
    (429, "model_rate_limited"), (500, "model_unavailable")])
def test_upstream_failures_are_honest_and_redacted(status, expected):
    async def scenario():
        def handle(request):
            return httpx.Response(status, json={"error": {"message": "raw-private-text-and-secret"}})
        provider = OpenAIProvider(Settings(provider="openai", model="test", api_key="test-key", cloud_acknowledged=True))
        await provider.client.close()
        provider.client = AsyncOpenAI(api_key="test-key", base_url="https://example.invalid", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        with pytest.raises(ProviderFailure) as exc:
            await provider.generate([{"role": "user", "content": "synthetic"}])
        assert exc.value.code == expected
        assert "raw-private" not in exc.value.message
        await provider.close()
    asyncio.run(scenario())


def test_core_returns_real_usage_and_does_not_fallback_to_demo(tmp_path):
    class RealResult:
        async def generate(self, messages):
            return Generation("针对当前内容生成的回答。", "tested-model", 100, 20)
        async def close(self):
            pass
    core = make_core(tmp_path, RealResult())
    result = run(core.chat("alice", ChatInput(text="synthetic")))
    assert result.model == "tested-model"
    assert result.input_tokens == 100 and result.output_tokens == 20
    run(core.close())


def test_sdk_accepts_tool_calls_and_returns_structured_arguments():
    async def scenario():
        def handle(request):
            payload = json.loads(request.content)
            assert payload['tools'][0]['function']['name'] == 'create_note'
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 1,
                'model': 'deepseek-flash', 'choices': [{'index': 0, 'finish_reason': 'tool_calls',
                'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_1',
                'type': 'function', 'function': {'name': 'create_note', 'arguments': '{"title":"A","content":"B"}'}}]}}],
                'usage': {'prompt_tokens': 11, 'completion_tokens': 4, 'total_tokens': 15}})
        provider = OpenAIProvider(Settings(provider='deepseek', model='deepseek-flash', api_key='test-key', cloud_acknowledged=True))
        await provider.client.close()
        provider.client = AsyncOpenAI(api_key='test-key', base_url='https://example.invalid', max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
        result = await provider.generate_with_tools([{'role': 'user', 'content': '写笔记'}],
            [{'type': 'function', 'function': {'name': 'create_note', 'parameters': {'type': 'object'}}}])
        assert result.tool_calls[0]['function']['name'] == 'create_note'
        assert result.input_tokens == 11
        await provider.close()
    asyncio.run(scenario())


def test_deepseek_search_uses_existing_key_and_returns_links(tmp_path, monkeypatch):
    import serein.core as core_module
    recorded = []
    def handle(request):
        recorded.append(request)
        assert request.url.host == 'api.deepseek.com'
        assert request.url.path == '/v1/tools/search'
        assert request.headers['Authorization'] == 'Bearer synthetic-key'
        assert json.loads(request.content)['query'] == '开源项目'
        return httpx.Response(200, json={'results': [{'title': '项目', 'url': 'https://example.org', 'snippet': '简介'}]})
    original_client = httpx.AsyncClient
    monkeypatch.setattr(core_module.httpx, 'AsyncClient',
                        lambda **kwargs: original_client(transport=httpx.MockTransport(handle)))
    core = make_core(tmp_path)
    core.settings.provider = 'deepseek'
    core.settings.api_key = 'synthetic-key'
    result = run(core.execute_tool('alice', 'web_search', {'query': '开源项目'}))
    assert result['ok'] and result['provider'] == 'deepseek'
    assert result['results'][0]['url'] == 'https://example.org'
    assert 'synthetic-key' not in str(result)
    assert len(recorded) == 1
    run(core.close())
