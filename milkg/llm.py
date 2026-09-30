"""Minimal OpenAI-compatible chat client; extraction and generation endpoints stay separate."""

import json
import re
import urllib.error
import urllib.request
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse


def is_local_endpoint(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}


class ChatModel(Protocol):
    def complete(self, system: str, user: str, temperature: float) -> dict: ...


def parse_json_object(text: str) -> dict:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("Model output has no JSON object") from None
        value, _ = decoder.raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("Model output must be a JSON object")
    return value


@dataclass
class OpenAICompatibleModel:
    base_url: str
    model: str
    api_key: str = ""
    timeout: int = 90
    json_mode: bool = True
    enable_thinking: bool | None = None
    reasoning_effort: str | None = None
    max_tokens: int | None = None

    def complete(self, system: str, user: str, temperature: float) -> dict:
        url = self.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        body = {
            "model": self.model,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.enable_thinking is not None:
            body["enable_thinking"] = self.enable_thinking
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(2):
            request = urllib.request.Request(url, data=payload, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                choice = result["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise RuntimeError(f"{self.model} 输出达到 max_tokens={self.max_tokens}；"
                                       "请提高本地模型最大输出 token，或减小分块字符数")
                content = choice["message"]["content"]
                return parse_json_object(content)
            except urllib.error.HTTPError as exc:
                detail = exc.read(500).decode("utf-8", "replace")
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 1:
                    raise RuntimeError(f"{self.model} HTTP {exc.code}: {detail}") from exc
            except (TimeoutError, urllib.error.URLError, ValueError, KeyError, TypeError) as exc:
                if attempt == 1:
                    raise RuntimeError(f"{self.model} response failed after 2 attempts: {type(exc).__name__}") from exc
            time.sleep(2 ** attempt)
        raise AssertionError("unreachable")
