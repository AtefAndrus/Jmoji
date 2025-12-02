"""OpenRouter chat completionクライアント.

公式ドキュメント: https://openrouter.ai/docs#chat-completions
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import httpx
from tenacity import Retrying, stop_after_attempt, wait_exponential


class OpenRouterClient:
    """OpenRouterの同期クライアント。

    - APIキーは環境変数`OPENROUTER_API_KEY`から取得（未設定なら例外）
    - 追加ヘッダー`HTTP-Referer`/`X-Title`はそれぞれ`OPENROUTER_HTTP_REFERER`、
      `OPENROUTER_TITLE`で指定（公式推奨）
    - チャット補完API `/chat/completions` を叩くシンプルなラッパ
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "anthropic/claude-haiku-4.5",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        self._client = httpx.Client(timeout=self.timeout)

    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        if referer:
            headers["HTTP-Referer"] = referer
        title = os.getenv("OPENROUTER_TITLE")
        if title:
            headers["X-Title"] = title
        return headers

    def _payload(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra:
            payload.update(extra)
        return payload

    # ------------------------------------------------------------------
    def _request(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """内部リクエスト処理（リトライなし）。"""
        payload = self._payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 100,
        model: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """テキストプロンプトを1ターンのmessagesに包んで送信。"""
        messages = [{"role": "user", "content": prompt}]
        for attempt in Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=60),
        ):
            with attempt:
                return self._request(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra=extra,
                )
        raise RuntimeError("Unreachable")

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """messagesをそのまま送るラッパ。"""
        for attempt in Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=60),
        ):
            with attempt:
                return self._request(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra=extra,
                )
        raise RuntimeError("Unreachable")

    # ------------------------------------------------------------------
    def stream(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Iterable[str]:
        """ストリーミングレスポンスをyield。

        OpenRouterはOpenAI互換のチャンクで`choices[].delta.content`を返す。
        """
        payload = self._payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra={"stream": True, **(extra or {})},
        )
        with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
        ) as r:
            r.raise_for_status()
            for text in r.iter_lines():
                if not text:
                    continue
                if text == "data: [DONE]":
                    continue
                if text.startswith("data: "):
                    text = text[len("data: ") :]

                try:
                    chunk = json.loads(text)
                    delta = chunk["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
                except Exception:
                    continue

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class AsyncOpenRouterClient:
    """OpenRouterの非同期クライアント。

    並列リクエストに対応。Semaphoreで同時リクエスト数を制御。
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "anthropic/claude-haiku-4.5",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
        max_retries: int = 3,
        max_concurrent: int = 5,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent

        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore: Optional["asyncio.Semaphore"] = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        """遅延初期化でAsyncClientを取得"""
        if self._client is None:
            import asyncio

            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._client

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        if referer:
            headers["HTTP-Referer"] = referer
        title = os.getenv("OPENROUTER_TITLE")
        if title:
            headers["X-Title"] = title
        return headers

    def _payload(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra:
            payload.update(extra)
        return payload

    async def _request_with_retry(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """リトライ付き非同期リクエスト"""
        import asyncio

        client = await self._ensure_client()
        assert self._semaphore is not None

        payload = self._payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            async with self._semaphore:
                try:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers(),
                        json=payload,
                    )

                    # レート制限対応
                    if response.status_code == 429:
                        retry_after = int(response.headers.get("retry-after", 10))
                        await asyncio.sleep(retry_after)
                        continue

                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"]

                except httpx.HTTPStatusError as e:
                    last_error = e
                    if e.response.status_code == 429:
                        retry_after = int(e.response.headers.get("retry-after", 10))
                        await asyncio.sleep(retry_after)
                    else:
                        # 指数バックオフ
                        wait_time = min(4 * (2**attempt), 60)
                        await asyncio.sleep(wait_time)
                except Exception as e:
                    last_error = e
                    wait_time = min(4 * (2**attempt), 60)
                    await asyncio.sleep(wait_time)

        raise last_error or RuntimeError("Max retries exceeded")

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 100,
        model: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """テキストプロンプトを1ターンのmessagesに包んで送信（非同期）"""
        messages = [{"role": "user", "content": prompt}]
        return await self._request_with_retry(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """messagesをそのまま送るラッパ（非同期）"""
        return await self._request_with_retry(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "AsyncOpenRouterClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


__all__ = ["OpenRouterClient", "AsyncOpenRouterClient"]
