"""Tiny LLM client – single function `chat`.

Usage:
    from llm import chat
    text = chat(prompt="서울 날씨 알려줘")
    raw  = chat(prompt="테스트", return_raw=True)
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import requests
import json
from typing import Type, TypeVar

# Optional pydantic support for structured output
try:
    from pydantic import BaseModel, ValidationError  # type: ignore
    _HAS_PYDANTIC = True
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore
    ValidationError = Exception  # type: ignore
    _HAS_PYDANTIC = False

# Defaults can be overridden via env vars
DEFAULT_API_URL = os.getenv("LLM_API_URL", "http://127.0.0.1:8080/v1/chat/completions")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen3-4b-q4_k_m")
DEFAULT_SYSTEM = os.getenv("LLM_SYSTEM", "You are a helpful assistant. Answer in Korean.")

__all__ = ["chat", "chat_structured", "LocalLLM", "DEFAULT_API_URL", "DEFAULT_MODEL"]

Message = Dict[str, str]


class LLMError(RuntimeError):
    """Raised when the LLM server returns a non-200 or an unexpected payload."""


def _convert_langchain_messages(messages: List[Message]) -> List[Message]:
    """Convert LangChain messages (HumanMessage/SystemMessage/AIMessage) or dicts
    into OpenAI-compatible dicts with {"role", "content"}.
    """
    converted: List[Message] = []
    for m in messages:
        if isinstance(m, dict) and "role" in m and "content" in m:
            converted.append({"role": str(m["role"]), "content": str(m["content"])})
            continue
        # LangChain messages expose `.type` ("human"|"ai"|"system"|"tool") and `.content`
        role = getattr(m, "type", None) or getattr(m, "role", None)
        content = getattr(m, "content", None)
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        elif role == "system":
            role = "system"
        elif role == "tool":
            role = "tool"
        if role and content is not None:
            converted.append({"role": str(role), "content": str(content)})
        else:
            raise ValueError(
                "Unsupported message object. Provide dicts or LangChain HumanMessage/SystemMessage/AIMessage."
            )
    return converted


def _messages(
    prompt: Optional[str],
    messages: Optional[List[Message]],
    system: str,
    no_think: bool,
) -> List[Message]:
    """Build a messages list from either a prompt or an explicit messages array."""
    if messages:
        return _convert_langchain_messages(messages)
    if not prompt:
        raise ValueError("Provide either `prompt` or `messages`.")
    if no_think and "/no_think" not in system:
        system = f"{system} /no_think"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def _strip_think(text: str) -> str:
    """Remove optional <think>...</think> wrapper if present."""
    if text.startswith("<think>"):
        parts = text.split("</think>", 1)
        if len(parts) == 2:
            return parts[1].strip()
    return text


T = TypeVar("T")

def _schema_to_json_schema(schema: object) -> Optional[Dict[str, Any]]:
    try:
        if _HAS_PYDANTIC and isinstance(schema, type) and issubclass(schema, BaseModel):
            if hasattr(schema, "model_json_schema"):
                return schema.model_json_schema()  # pydantic v2
            if hasattr(schema, "schema"):
                return schema.schema()  # pydantic v1
    except Exception:
        return None
    return None


def _extract_json_obj(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from text and load it."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    return json.loads(text)


def chat(
    prompt: Optional[str] = None,
    messages: Optional[List[Message]] = None,
    *,
    model: str = DEFAULT_MODEL,
    api_url: str = DEFAULT_API_URL,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    reasoning: bool = False,
    logprobs: Optional[Dict[str, Any]] = None,
    system: str = DEFAULT_SYSTEM,
    timeout: Optional[float] = None,
    strip_think_blocks: bool = True,
    return_raw: bool = False,
    headers: Optional[Dict[str, str]] = None,
) -> Union[str, Dict[str, Any]]:
    """Minimal, readable LLM call.

    Returns assistant text by default; set `return_raw=True` to get an OpenAI-like dict.
    """
    msgs = _messages(prompt, messages, system, no_think=not reasoning)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    if not reasoning:
        payload["reasoning"] = {"budget": 0, "format": "none"}
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    if logprobs is not None:
        payload["logprobs"] = logprobs

    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    start = time.time()
    resp = requests.post(api_url, json=payload, headers=req_headers, timeout=timeout)
    latency = time.time() - start

    if resp.status_code != 200:
        raise LLMError(f"{resp.status_code}: {resp.text}")

    data = resp.json()
    envelope: Dict[str, Any] = {
        "id": data.get("id", str(uuid.uuid4())),
        "object": data.get("object", "chat.completion"),
        "created": data.get("created", int(time.time())),
        "model": data.get("model", model),
        "choices": data.get("choices", []),
        "usage": data.get("usage", {}),
        "latency": latency,
    }

    if return_raw:
        envelope["_payload"] = payload
        return envelope

    try:
        text = envelope["choices"][0]["message"]["content"]
    except Exception as exc:
        raise LLMError(f"Unexpected response format. Raw: {data}") from exc

    return _strip_think(text) if strip_think_blocks else text


def chat_structured(
    *,
    schema: Optional[Type[T]] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    prompt: Optional[str] = None,
    messages: Optional[List[Message]] = None,
    model: str = DEFAULT_MODEL,
    api_url: str = DEFAULT_API_URL,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    reasoning: bool = False,
    system: str = DEFAULT_SYSTEM,
    timeout: Optional[float] = None,
    headers: Optional[Dict[str, str]] = None,
    strict_json: bool = True,
) -> Union[T, Dict[str, Any]]:
    """Call the LLM and parse **structured JSON** according to a Pydantic model or JSON schema.

    Returns a Pydantic model instance when `schema` is provided (and pydantic is installed),
    otherwise returns a dict.
    """
    # 1) Normalize messages or build from prompt
    base_msgs = _messages(prompt, messages, system, no_think=not reasoning)

    # 2) Inject a JSON-only formatting instruction as a leading system message
    schema_dict = json_schema or _schema_to_json_schema(schema)
    guide = "반드시 순수 JSON 객체만 출력하세요. 마크다운 코드블록 없이 한 개의 JSON만 반환하세요."
    if schema_dict:
        guide += "\n다음 JSON 스키마를 따르세요:\n" + json.dumps(schema_dict, ensure_ascii=False)
    injected = [{"role": "system", "content": guide}] + base_msgs

    # 3) Call the base chat function
    text = chat(
        messages=injected,
        model=model,
        api_url=api_url,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        reasoning=reasoning,
        system=system,
        timeout=timeout,
        headers=headers,
        strip_think_blocks=True,
        return_raw=False,
    )

    # 4) Extract and validate JSON
    obj = _extract_json_obj(text) if strict_json else json.loads(text)

    if schema and _HAS_PYDANTIC and isinstance(schema, type) and issubclass(schema, BaseModel):
        try:
            # pydantic v2
            if hasattr(schema, "model_validate"):
                return schema.model_validate(obj)
            # pydantic v1
            return schema.parse_obj(obj)  # type: ignore[attr-defined]
        except ValidationError as e:  # pragma: no cover
            raise LLMError(f"Structured validation failed: {e}")

    return obj


class LocalLLM:
    """경량 어댑터: `.with_structured_output()` 패턴과 인스턴스 메소드 `.chat()`/`.chat_structured()` 제공.

    Example:
        from llm import LocalLLM
        from langchain_core.messages import HumanMessage, SystemMessage
        from pydantic import BaseModel
        from typing import List

        class Categories(BaseModel):
            categories: List[str]

        llm = LocalLLM()

        # 1) 간단 호춣 (텍스트)
        txt = llm.chat(messages=[
            SystemMessage(content="너는 한국어로만 답해."),
            HumanMessage(content="안녕!"),
        ])

        # 2) 구조화된 출력
        router_chain = llm.with_structured_output(Categories)
        resp = router_chain.invoke([
            SystemMessage(content="JSON으로만 {\"categories\":[...]} 를 반환하라."),
            HumanMessage(content="비밀번호 재설정이 필요해요"),
        ])
        print(resp.categories)
    """

    def __init__(self, *, model: str = DEFAULT_MODEL, api_url: str = DEFAULT_API_URL, **defaults: Any) -> None:
        self.model = model
        self.api_url = api_url
        self.defaults = defaults

    def chat(self,
             prompt: Optional[str] = None,
             messages: Optional[List[Message]] = None,
             **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """인스턴스 기본값(self.model, self.api_url, self.defaults)을 적용하여 채팅 호출."""
        params: Dict[str, Any] = {"model": self.model, "api_url": self.api_url}
        params.update(self.defaults)
        params.update(kwargs)
        return chat(prompt=prompt, messages=messages, **params)

    def chat_structured(self,
                        *,
                        schema: Optional[Type[T]] = None,
                        json_schema: Optional[Dict[str, Any]] = None,
                        prompt: Optional[str] = None,
                        messages: Optional[List[Message]] = None,
                        **kwargs: Any) -> Union[T, Dict[str, Any]]:
        """인스턴스 기본값을 적용하여 구조화된(JSON) 출력을 반환."""
        params: Dict[str, Any] = {"model": self.model, "api_url": self.api_url}
        params.update(self.defaults)
        params.update(kwargs)
        return chat_structured(schema=schema, json_schema=json_schema, prompt=prompt, messages=messages, **params)

    def with_structured_output(self, schema: Type[T]):
        class _StructuredCaller:
            def __init__(self, outer: "LocalLLM", schema_: Type[T]) -> None:
                self.outer = outer
                self.schema = schema_

            def invoke(self, messages: List[Message]) -> T:
                return chat_structured(
                    schema=self.schema,
                    messages=messages,
                    model=self.outer.model,
                    api_url=self.outer.api_url,
                    **self.outer.defaults,
                )

        return _StructuredCaller(self, schema)

