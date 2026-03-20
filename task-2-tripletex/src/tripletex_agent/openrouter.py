import json
from typing import Any

import httpx

from tripletex_agent.config import Settings


class OpenRouterError(Exception):
    pass


class OpenRouterClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.openrouter_base_url.rstrip("/"),
            timeout=settings.http_timeout_seconds,
            headers=self._build_headers(),
        )

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "X-Title": self._settings.openrouter_app_name,
        }
        if self._settings.openrouter_site_url:
            headers["HTTP-Referer"] = self._settings.openrouter_site_url
        return headers

    async def close(self) -> None:
        await self._client.aclose()

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int = 1500,
    ) -> dict[str, Any]:
        if not self._settings.openrouter_api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not configured")

        payload: dict[str, Any] = {
            "model": self._settings.openrouter_model,
            "messages": messages,
            "temperature": self._settings.agent_model_temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format

        response = await self._client.post("/chat/completions", json=payload)
        if response.status_code >= 400:
            raise OpenRouterError(self._build_error_message(response))

        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise OpenRouterError("OpenRouter returned no choices")

        message = choices[0].get("message")
        if not message:
            raise OpenRouterError("OpenRouter returned no message")
        return message

    async def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int = 800,
    ) -> dict[str, Any]:
        message = await self.chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=0.0,
        )
        content = message.get("content")
        if not isinstance(content, str):
            raise OpenRouterError("Expected JSON content string from OpenRouter")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise OpenRouterError(f"Failed to parse JSON planner output: {exc}") from exc

    def _build_error_message(self, response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            body = response.text
        return f"OpenRouter request failed with status {response.status_code}: {body}"
