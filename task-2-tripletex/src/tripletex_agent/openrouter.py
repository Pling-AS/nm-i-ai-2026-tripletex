import asyncio
import json
import logging
import re
import subprocess
import time
from typing import Any

import httpx

from tripletex_agent.config import Settings

logger = logging.getLogger(__name__)


class OpenRouterError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _is_anthropic_model(model: str) -> bool:
    return "claude" in model.lower()


_VERTEX_MODEL_ALIASES: dict[str, str] = {
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-opus-4-6": "claude-opus-4-6",
    "claude-sonnet-4-20250514": "claude-sonnet-4-6",
    "claude-opus-4-20250514": "claude-opus-4-6",
}


def _normalize_claude_model_for_vertex(model: str) -> str | None:
    cleaned = model.lower().strip()
    cleaned = re.sub(r"^anthropic/", "", cleaned)
    cleaned = re.sub(r":.*$", "", cleaned)
    cleaned = cleaned.replace(".", "-")

    if cleaned in _VERTEX_MODEL_ALIASES:
        return _VERTEX_MODEL_ALIASES[cleaned]

    for alias, vertex_id in _VERTEX_MODEL_ALIASES.items():
        if alias in cleaned:
            return vertex_id

    return None


# ---------------------------------------------------------------------------
# Message / tool format translation: OpenAI <-> Anthropic
# ---------------------------------------------------------------------------


def _openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        fn = tool.get("function", tool)
        result.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
        )
    return result


def _openai_messages_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    anthropic_msgs: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                system_parts.append(
                    "\n".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    )
                )
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                content_blocks: list[dict[str, Any]] = []
                text = msg.get("content", "")
                if text:
                    content_blocks.append({"type": "text", "text": text})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    raw_args = fn.get("arguments", "{}")
                    try:
                        input_data = (
                            json.loads(raw_args)
                            if isinstance(raw_args, str)
                            else raw_args
                        )
                    except json.JSONDecodeError:
                        input_data = {}
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": input_data,
                        }
                    )
                anthropic_msgs.append({"role": "assistant", "content": content_blocks})
            else:
                content = msg.get("content", "")
                if isinstance(content, list):
                    anthropic_msgs.append({"role": "assistant", "content": content})
                else:
                    anthropic_msgs.append(
                        {"role": "assistant", "content": content or ""}
                    )
            continue

        if role == "tool":
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            if (
                anthropic_msgs
                and anthropic_msgs[-1].get("role") == "user"
                and isinstance(anthropic_msgs[-1].get("content"), list)
                and anthropic_msgs[-1]["content"]
                and isinstance(anthropic_msgs[-1]["content"][0], dict)
                and anthropic_msgs[-1]["content"][0].get("type") == "tool_result"
            ):
                anthropic_msgs[-1]["content"].append(tool_result_block)
            else:
                anthropic_msgs.append({"role": "user", "content": [tool_result_block]})
            continue

        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                anthropic_content: list[dict[str, Any]] = []
                for part in content:
                    if isinstance(part, dict):
                        ptype = part.get("type", "")
                        if ptype == "text":
                            anthropic_content.append(
                                {"type": "text", "text": part.get("text", "")}
                            )
                        elif ptype == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                match = re.match(
                                    r"data:([^;]+);base64,(.+)", url, re.DOTALL
                                )
                                if match:
                                    anthropic_content.append(
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": match.group(1),
                                                "data": match.group(2),
                                            },
                                        }
                                    )
                            else:
                                anthropic_content.append(
                                    {
                                        "type": "image",
                                        "source": {"type": "url", "url": url},
                                    }
                                )
                        else:
                            anthropic_content.append(part)
                    else:
                        anthropic_content.append({"type": "text", "text": str(part)})
                anthropic_msgs.append({"role": "user", "content": anthropic_content})
            else:
                anthropic_msgs.append({"role": "user", "content": content or ""})
            continue

        anthropic_msgs.append({"role": role, "content": msg.get("content", "")})

    return "\n\n".join(system_parts), anthropic_msgs


def _anthropic_response_to_openai(response_data: dict[str, Any]) -> dict[str, Any]:
    content_blocks = response_data.get("content", [])
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in content_blocks:
        btype = block.get("type", "")
        if btype == "thinking":
            thinking_parts.append(block.get("thinking", ""))
        elif btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(
                            block.get("input", {}), ensure_ascii=False
                        ),
                    },
                }
            )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else "",
    }
    if thinking_parts:
        message["thinking"] = "\n".join(thinking_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OpenRouterClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        # OpenAI-compatible client (OpenRouter or Azure OpenAI)
        if settings.azure_api_key:
            self._openai_client = httpx.AsyncClient(
                base_url=settings.azure_openai_base_url.rstrip("/"),
                timeout=settings.http_timeout_seconds,
                headers={
                    "api-key": settings.azure_api_key,
                    "Content-Type": "application/json",
                },
            )
        else:
            self._openai_client = httpx.AsyncClient(
                base_url=settings.openrouter_base_url.rstrip("/"),
                timeout=settings.http_timeout_seconds,
                headers=self._build_openrouter_headers(),
            )

        # Azure Anthropic client (fallback for Claude models)
        self._azure_anthropic_client: httpx.AsyncClient | None = None
        if settings.azure_api_key and settings.azure_anthropic_base_url:
            self._azure_anthropic_client = httpx.AsyncClient(
                base_url=settings.azure_anthropic_base_url.rstrip("/"),
                timeout=settings.http_timeout_seconds,
                headers={
                    "x-api-key": settings.azure_api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
            )

        # Vertex AI client (primary for Claude models)
        self._vertex_client: httpx.AsyncClient | None = None
        self._vertex_token: str = ""
        self._vertex_token_expiry: float = 0.0
        self._vertex_token_lock: asyncio.Lock = asyncio.Lock()
        if settings.vertex_ai_enabled and settings.vertex_ai_project_id:
            self._vertex_client = httpx.AsyncClient(
                timeout=settings.http_timeout_seconds,
                headers={"Content-Type": "application/json"},
            )

    def _build_openrouter_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "X-Title": self._settings.openrouter_app_name,
        }
        if self._settings.openrouter_site_url:
            headers["HTTP-Referer"] = self._settings.openrouter_site_url
        return headers

    async def close(self) -> None:
        await self._openai_client.aclose()
        if self._azure_anthropic_client:
            await self._azure_anthropic_client.aclose()
        if self._vertex_client:
            await self._vertex_client.aclose()

    # ------ Vertex AI token management ------

    async def _get_vertex_access_token(self, *, force_refresh: bool = False) -> str:
        async with self._vertex_token_lock:
            now = time.monotonic()
            if (
                not force_refresh
                and self._vertex_token
                and now < self._vertex_token_expiry
            ):
                return self._vertex_token

            token = await asyncio.to_thread(self._fetch_gcloud_token)
            self._vertex_token = token
            self._vertex_token_expiry = now + self._settings.vertex_ai_token_ttl_seconds
            return token

    def _fetch_gcloud_token(self) -> str:
        result = subprocess.run(
            [self._settings.vertex_ai_gcloud_bin, "auth", "print-access-token"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise OpenRouterError(
                f"gcloud token fetch failed (exit {result.returncode}): {result.stderr.strip()}",
            )
        token = result.stdout.strip()
        if not token:
            raise OpenRouterError("gcloud returned empty access token")
        return token

    def _build_vertex_url(self, vertex_model: str) -> str:
        s = self._settings
        return (
            f"https://{s.vertex_ai_region}-aiplatform.googleapis.com/v1/"
            f"projects/{s.vertex_ai_project_id}/locations/{s.vertex_ai_region}/"
            f"publishers/anthropic/models/{vertex_model}:rawPredict"
        )

    # ------ Public interface ------

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int = 1500,
        model_override: str | None = None,
        enable_thinking: bool = False,
    ) -> dict[str, Any]:
        model = model_override or self._settings.openrouter_model
        temp = (
            self._settings.agent_model_temperature
            if temperature is None
            else temperature
        )

        if _is_anthropic_model(model):
            return await self._claude_chat_completion(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temp,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
        return await self._openai_chat_completion(
            model=model,
            messages=messages,
            tools=tools,
            response_format=response_format,
            temperature=temp,
            max_tokens=max_tokens,
        )

    async def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int = 1200,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        model = model_override or self._settings.openrouter_model

        if _is_anthropic_model(model):
            message = await self._claude_chat_completion(
                model=model,
                messages=messages,
                tools=None,
                temperature=0.0,
                max_tokens=max_tokens,
            )
        else:
            message = await self._openai_chat_completion(
                model=model,
                messages=messages,
                tools=None,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=max_tokens,
            )

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            message = await self.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                model_override=model,
            )
            content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise OpenRouterError(f"LLM returned empty content. Raw message: {message}")
        return self._extract_json(content)

    # ------ Claude routing: Vertex AI primary, Azure fallback ------

    async def _claude_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
    ) -> dict[str, Any]:
        vertex_model = _normalize_claude_model_for_vertex(model)
        vertex_available = self._vertex_client is not None and vertex_model is not None

        if vertex_available:
            assert self._vertex_client is not None
            assert vertex_model is not None
            try:
                return await self._vertex_anthropic_chat_completion(
                    vertex_model=vertex_model,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                )
            except OpenRouterError as vertex_exc:
                sc = vertex_exc.status_code
                if sc == 400:
                    raise

                if sc in (401, 403):
                    try:
                        return await self._vertex_anthropic_chat_completion(
                            vertex_model=vertex_model,
                            messages=messages,
                            tools=tools,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            force_token_refresh=True,
                            enable_thinking=enable_thinking,
                        )
                    except Exception:
                        pass

                if self._azure_anthropic_client:
                    logger.warning(
                        "Vertex AI failed (status=%s), falling back to Azure: %s",
                        sc,
                        vertex_exc,
                    )
                    return await self._azure_anthropic_chat_completion(
                        model=model,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        enable_thinking=enable_thinking,
                    )
                raise

            except (httpx.TimeoutException, httpx.TransportError) as vertex_exc:
                if self._azure_anthropic_client:
                    logger.warning(
                        "Vertex AI transport error, falling back to Azure: %s",
                        vertex_exc,
                    )
                    return await self._azure_anthropic_chat_completion(
                        model=model,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        enable_thinking=enable_thinking,
                    )
                raise OpenRouterError(
                    f"Vertex AI transport error: {vertex_exc}"
                ) from vertex_exc

        if self._azure_anthropic_client:
            return await self._azure_anthropic_chat_completion(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )

        raise OpenRouterError(
            "No Claude backend available: Vertex AI not configured and Azure Anthropic not configured"
        )

    # ------ Vertex AI Anthropic backend (primary) ------

    async def _vertex_anthropic_chat_completion(
        self,
        *,
        vertex_model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        force_token_refresh: bool = False,
        enable_thinking: bool = False,
    ) -> dict[str, Any]:
        if not self._vertex_client:
            raise OpenRouterError("Vertex AI client not configured")

        token = await self._get_vertex_access_token(
            force_refresh=force_token_refresh,
        )

        system_prompt, anthropic_messages = _openai_messages_to_anthropic(messages)

        thinking_budget = 0
        if enable_thinking:
            thinking_budget = min(max_tokens - 2048, 50000)
            if thinking_budget < 1024:
                thinking_budget = 1024
                max_tokens = max(max_tokens, 4096)

        payload: dict[str, Any] = {
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "temperature": 1 if enable_thinking else temperature,
        }
        if enable_thinking:
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = _openai_tools_to_anthropic(tools)
            payload["tool_choice"] = {"type": "auto"}

        url = self._build_vertex_url(vertex_model)
        response = await self._vertex_client.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        if response.status_code >= 400:
            raise OpenRouterError(
                self._build_error_message(response),
                status_code=response.status_code,
            )

        body = response.json()
        if body.get("type") == "error":
            err = body.get("error", {})
            raise OpenRouterError(
                f"Vertex Anthropic error: {err.get('type', 'unknown')}: {err.get('message', '')}",
                status_code=response.status_code,
            )

        logger.debug("Claude request served by provider=vertex model=%s", vertex_model)
        return _anthropic_response_to_openai(body)

    # ------ Azure Anthropic backend (fallback) ------

    async def _azure_anthropic_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
    ) -> dict[str, Any]:
        if not self._azure_anthropic_client:
            raise OpenRouterError("Azure Anthropic client not configured")

        system_prompt, anthropic_messages = _openai_messages_to_anthropic(messages)

        thinking_budget = 0
        if enable_thinking:
            thinking_budget = min(max_tokens - 2048, 50000)
            if thinking_budget < 1024:
                thinking_budget = 1024
                max_tokens = max(max_tokens, 4096)

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "temperature": 1 if enable_thinking else temperature,
        }
        if enable_thinking:
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = _openai_tools_to_anthropic(tools)
            payload["tool_choice"] = {"type": "auto"}

        response = await self._azure_anthropic_client.post("/messages", json=payload)
        if response.status_code >= 400:
            raise OpenRouterError(
                self._build_error_message(response),
                status_code=response.status_code,
            )

        body = response.json()
        if body.get("type") == "error":
            err = body.get("error", {})
            raise OpenRouterError(
                f"Anthropic error: {err.get('type', 'unknown')}: {err.get('message', '')}",
                status_code=response.status_code,
            )

        logger.debug(
            "Claude request served by provider=azure_anthropic model=%s", model
        )
        return _anthropic_response_to_openai(body)

    # ------ OpenAI-compatible backend ------

    async def _openai_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1500,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if "gpt-5" in model or "o3" in model or "o4" in model:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format

        response = await self._openai_client.post("/chat/completions", json=payload)
        if response.status_code >= 400:
            raise OpenRouterError(
                self._build_error_message(response),
                status_code=response.status_code,
            )

        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise OpenRouterError("OpenAI returned no choices")
        message = choices[0].get("message")
        if not message:
            raise OpenRouterError("OpenAI returned no message")
        return message

    # ------ Shared helpers ------

    def _extract_json(self, content: str) -> dict[str, Any]:
        stripped = content.strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        code_block_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```",
            stripped,
            re.DOTALL,
        )
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        brace_match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        raise OpenRouterError(f"Failed to parse JSON from LLM output: {content[:300]}")

    def _build_error_message(self, response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            body = response.text
        return f"LLM request failed with status {response.status_code}: {body}"
