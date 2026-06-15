from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from app.core.config import settings


VALID_JSON_ESCAPE_CHARS = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}


def _get_setting(name: str, default: Any = None) -> Any:
    return getattr(settings, name, os.getenv(name, default))


def _repair_json_string_escapes(content: str) -> str:
    """Repair common invalid JSON emitted when LLM strings contain LaTeX."""

    output: list[str] = []
    in_string = False
    i = 0

    while i < len(content):
        ch = content[i]

        if not in_string:
            output.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue

        if ch == '"':
            output.append(ch)
            in_string = False
            i += 1
            continue

        if ch == "\\":
            if i + 1 >= len(content):
                output.append("\\\\")
                i += 1
                continue

            next_ch = content[i + 1]
            next_next_ch = content[i + 2] if i + 2 < len(content) else ""
            if next_ch in {"b", "f", "n", "r", "t"} and next_next_ch.isalpha():
                # JSON would treat \frac, \tan, \theta, \nabla, \bar as
                # control-character escapes. They are almost certainly LaTeX.
                output.append("\\\\")
                output.append(next_ch)
                i += 2
                continue

            if next_ch in VALID_JSON_ESCAPE_CHARS:
                output.append(ch)
                output.append(next_ch)
            else:
                # JSON does not allow escapes like \(, \[, \frac, \sin.
                # Preserve the intended LaTeX by escaping the backslash itself.
                output.append("\\\\")
                output.append(next_ch)
            i += 2
            continue

        if ch == "\n":
            output.append("\\n")
            i += 1
            continue

        if ch == "\r":
            output.append("\\n")
            if i + 1 < len(content) and content[i + 1] == "\n":
                i += 2
            else:
                i += 1
            continue

        if ch == "\t":
            output.append("\\t")
            i += 1
            continue

        if ord(ch) < 0x20:
            output.append(" ")
            i += 1
            continue

        output.append(ch)
        i += 1

    return "".join(output)


def _loads_llm_json(content: str) -> dict[str, Any]:
    repaired = _repair_json_string_escapes(content)

    if repaired != content:
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            data = json.loads(content)
    else:
        data = json.loads(content)

    if not isinstance(data, dict):
        raise ValueError("模型返回的 JSON 不是对象")

    return data


@dataclass
class LLMResponse:
    content: str
    data: dict[str, Any]
    reasoning_content: str | None = None
    raw_response: Any | None = None


class LLMService:
    """DeepSeek OpenAI-compatible chat service."""

    def __init__(self) -> None:
        self.api_key: str = str(_get_setting("LLM_API_KEY", "")).strip()
        self.base_url: str = str(_get_setting("LLM_BASE_URL", "https://api.deepseek.com")).strip()
        self.model: str = str(_get_setting("LLM_MODEL", "deepseek-reasoner")).strip()
        self.timeout: int = int(_get_setting("LLM_TIMEOUT", 120))
        self.max_tokens: int = int(_get_setting("LLM_MAX_TOKENS", 2048))
        self.temperature: float = float(_get_setting("LLM_TEMPERATURE", 0.2))
        self.return_reasoning: bool = str(_get_setting("LLM_RETURN_REASONING", "false")).lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def chat_json(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        if not messages:
            raise ValueError("messages 不能为空")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            response_format={"type": "json_object"},
        )

        choice = response.choices[0]
        message = choice.message
        content = (message.content or "").strip()
        reasoning_content = getattr(message, "reasoning_content", None)
        finish_reason = getattr(choice, "finish_reason", None)

        if not content:
            raise RuntimeError("模型返回内容为空，finish_reason={finish_reason}")

        try:
            data = _loads_llm_json(content)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"模型未返回合法 JSON：{content[:800]}") from exc

        return LLMResponse(
            content=content,
            data=data,
            reasoning_content=reasoning_content if self.return_reasoning else None,
            raw_response=response,
        )


_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def chat_json(
    messages: Sequence[ChatCompletionMessageParam],
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> LLMResponse:
    return get_llm_service().chat_json(messages=messages, max_tokens=max_tokens, temperature=temperature)
