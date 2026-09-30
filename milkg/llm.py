"""Minimal OpenAI-compatible chat client; extraction and generation endpoints stay separate."""

import json
import http.client
import io
import re
import socket
import threading
import urllib.error
import urllib.request
import time
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse


def is_local_endpoint(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}


class ChatModel(Protocol):
    def complete(self, system: str, user: str, temperature: float) -> dict: ...


class RequestCancelled(Exception):
    """The user stopped the current job."""


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
    cancel_event: threading.Event | None = field(default=None, repr=False)
    _connection: http.client.HTTPConnection | None = field(default=None, init=False, repr=False)
    _connection_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def cancel(self) -> None:
        if self.cancel_event is not None:
            self.cancel_event.set()
        with self._connection_lock:
            connection = self._connection
        if connection is not None:
            try:
                if connection.sock is not None:
                    connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def _request(self, url: str, payload: bytes, headers: dict) -> dict:
        # The local Ollama connection can be shut down when a job is cancelled.
        if not (self.cancel_event is not None and is_local_endpoint(url)):
            request = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        if self.cancel_event.is_set():
            raise RequestCancelled()
        parsed = urlparse(url)
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=self.timeout)
        with self._connection_lock:
            self._connection = connection
        try:
            if self.cancel_event.is_set():
                raise RequestCancelled()
            path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            connection.request("POST", path, body=payload, headers=headers)
            response = connection.getresponse()
            data = response.read()
            if response.status >= 400:
                raise urllib.error.HTTPError(url, response.status, response.reason,
                                             response.headers, io.BytesIO(data))
            return json.loads(data)
        finally:
            with self._connection_lock:
                if self._connection is connection:
                    self._connection = None
            connection.close()

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
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise RequestCancelled()
            try:
                result = self._request(url, payload, headers)
                choice = result["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise RuntimeError(f"{self.model} 输出达到 max_tokens={self.max_tokens}；"
                                       "请提高本地模型最大输出 token，或减小分块字符数")
                content = choice["message"]["content"]
                return parse_json_object(content)
            except urllib.error.HTTPError as exc:
                if self.cancel_event is not None and self.cancel_event.is_set():
                    raise RequestCancelled() from exc
                detail = exc.read(500).decode("utf-8", "replace")
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 1:
                    raise RuntimeError(f"{self.model} HTTP {exc.code}: {detail}") from exc
            except (TimeoutError, OSError, urllib.error.URLError, ValueError, KeyError, TypeError) as exc:
                if self.cancel_event is not None and self.cancel_event.is_set():
                    raise RequestCancelled() from exc
                if attempt == 1:
                    raise RuntimeError(f"{self.model} response failed after 2 attempts: {type(exc).__name__}") from exc
            if self.cancel_event is not None:
                if self.cancel_event.wait(2 ** attempt):
                    raise RequestCancelled()
            else:
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")
