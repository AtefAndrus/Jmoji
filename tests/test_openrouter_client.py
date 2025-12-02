"""OpenRouterクライアントのテスト

実際のAPIは呼び出さず、モックを使用してテストする。
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.generation.openrouter_client import AsyncOpenRouterClient, OpenRouterClient


# =============================================================================
# OpenRouterClient（同期版）テスト
# =============================================================================


def test_sync_client_init_with_api_key():
    """APIキーを直接指定して初期化できる"""
    client = OpenRouterClient(api_key="test-key")
    assert client.api_key == "test-key"
    client.close()


def test_sync_client_init_from_env():
    """環境変数からAPIキーを取得できる"""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}):
        client = OpenRouterClient()
        assert client.api_key == "env-key"
        client.close()


def test_sync_client_init_raises_without_api_key():
    """APIキーがない場合はValueErrorを発生"""
    with patch.dict(os.environ, {}, clear=True):
        # OPENROUTER_API_KEYを削除
        os.environ.pop("OPENROUTER_API_KEY", None)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not set"):
            OpenRouterClient()


def test_sync_client_headers():
    """ヘッダーが正しく生成される"""
    client = OpenRouterClient(api_key="test-key")
    headers = client._headers()

    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"
    client.close()


def test_sync_client_headers_with_referer():
    """HTTP-Refererヘッダーが含まれる"""
    with patch.dict(os.environ, {"OPENROUTER_HTTP_REFERER": "https://example.com"}):
        client = OpenRouterClient(api_key="test-key")
        headers = client._headers()
        assert headers["HTTP-Referer"] == "https://example.com"
        client.close()


def test_sync_client_payload():
    """ペイロードが正しく生成される"""
    client = OpenRouterClient(api_key="test-key", model="test-model")
    messages = [{"role": "user", "content": "Hello"}]

    payload = client._payload(messages, temperature=0.5, max_tokens=50)

    assert payload["model"] == "test-model"
    assert payload["messages"] == messages
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 50
    client.close()


def test_sync_client_context_manager():
    """コンテキストマネージャとして使用できる"""
    with OpenRouterClient(api_key="test-key") as client:
        assert client.api_key == "test-key"


@patch("httpx.Client.post")
def test_sync_client_complete(mock_post):
    """completeメソッドが正しくAPIを呼び出す"""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello, World!"}}]
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    client = OpenRouterClient(api_key="test-key")
    result = client.complete("Say hello")

    assert result == "Hello, World!"
    mock_post.assert_called_once()
    client.close()


# =============================================================================
# AsyncOpenRouterClient（非同期版）テスト
# =============================================================================


def test_async_client_init_with_api_key():
    """APIキーを直接指定して初期化できる"""
    client = AsyncOpenRouterClient(api_key="test-key")
    assert client.api_key == "test-key"
    assert client.max_concurrent == 5


def test_async_client_init_custom_concurrent():
    """max_concurrentをカスタマイズできる"""
    client = AsyncOpenRouterClient(api_key="test-key", max_concurrent=10)
    assert client.max_concurrent == 10


def test_async_client_init_raises_without_api_key():
    """APIキーがない場合はValueErrorを発生"""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("OPENROUTER_API_KEY", None)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not set"):
            AsyncOpenRouterClient()


def test_async_client_headers():
    """ヘッダーが正しく生成される"""
    client = AsyncOpenRouterClient(api_key="test-key")
    headers = client._headers()

    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"


def test_async_client_payload():
    """ペイロードが正しく生成される"""
    client = AsyncOpenRouterClient(api_key="test-key", model="test-model")
    messages = [{"role": "user", "content": "Hello"}]

    payload = client._payload(messages, temperature=0.5, max_tokens=50)

    assert payload["model"] == "test-model"
    assert payload["messages"] == messages
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 50


@pytest.mark.asyncio
async def test_async_client_context_manager():
    """非同期コンテキストマネージャとして使用できる"""
    async with AsyncOpenRouterClient(api_key="test-key") as client:
        assert client.api_key == "test-key"


@pytest.mark.asyncio
async def test_async_client_complete():
    """completeメソッドが正しくAPIを呼び出す"""
    client = AsyncOpenRouterClient(api_key="test-key")

    # AsyncClientのモック
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello, World!"}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await client.complete("Say hello")

    assert result == "Hello, World!"
    await client.close()


@pytest.mark.asyncio
async def test_async_client_rate_limit_retry():
    """429エラー時にリトライする"""
    client = AsyncOpenRouterClient(api_key="test-key", max_retries=2)

    # 最初は429、次は成功
    mock_response_429 = MagicMock()
    mock_response_429.status_code = 429
    mock_response_429.headers = {"retry-after": "1"}

    mock_response_ok = MagicMock()
    mock_response_ok.status_code = 200
    mock_response_ok.json.return_value = {
        "choices": [{"message": {"content": "Success"}}]
    }
    mock_response_ok.raise_for_status = MagicMock()

    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_response_429
        return mock_response_ok

    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        result = await client.complete("Test")

    assert result == "Success"
    assert call_count == 2
    await client.close()


@pytest.mark.asyncio
async def test_async_client_semaphore_limits_concurrency():
    """Semaphoreが同時リクエスト数を制限する"""
    client = AsyncOpenRouterClient(api_key="test-key", max_concurrent=2)

    active_requests = 0
    max_active = 0

    async def mock_post(*args, **kwargs):
        nonlocal active_requests, max_active
        active_requests += 1
        max_active = max(max_active, active_requests)
        await asyncio.sleep(0.1)  # 短い遅延
        active_requests -= 1

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OK"}}]
        }
        mock_response.raise_for_status = MagicMock()
        return mock_response

    with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
        # 5つの並列リクエストを発行
        tasks = [client.complete(f"Test {i}") for i in range(5)]
        await asyncio.gather(*tasks)

    # max_concurrent=2 なので、最大同時実行数は2以下
    assert max_active <= 2
    await client.close()
