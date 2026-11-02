import json
import re
import urllib.request
from typing import Any


def _normalize_usage(usage: Any) -> dict[str, int] | None:
    """Normalize OpenAI-compatible usage fields returned by Qwen servers."""
    if not isinstance(usage, dict):
        return None
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        details = usage.get("output_tokens_details")
    if not isinstance(details, dict):
        details = {}
    aliases = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": ("completion_tokens", "output_tokens"),
        "total_tokens": ("total_tokens",),
        "reasoning_tokens": ("reasoning_tokens",),
    }
    normalized: dict[str, int] = {}
    for target, keys in aliases.items():
        for key in keys:
            value = details.get(key) if target == "reasoning_tokens" else usage.get(key)
            if value is None and target == "reasoning_tokens":
                value = usage.get(key)
            if isinstance(value, (int, float)):
                normalized[target] = int(value)
                break
    if "total_tokens" not in normalized:
        prompt = normalized.get("prompt_tokens", 0)
        completion = normalized.get("completion_tokens", 0)
        if prompt or completion:
            normalized["total_tokens"] = prompt + completion
    if "reasoning_tokens" in normalized and "completion_tokens" in normalized:
        normalized["visible_completion_tokens"] = max(
            0, normalized["completion_tokens"] - normalized["reasoning_tokens"]
        )
    return normalized or None


def chat_completion(
    messages: list[dict[str, Any]],
    model: str,
    api_key: str | None,
    base_url: str,
    temperature: float | None,
    max_tokens: int | None,
    timeout: int,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    '''
    构造请求、发送请求、解析响应，返回模型的回答
    '''
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if extra_body:
        payload.update(extra_body)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    choice = data["choices"][0]
    message = choice.get("message", {})
    # Providers use several names for hidden/reasoning output. Keep it
    # separate from the visible answer so it can be audited later.
    thinking = (
        message.get("reasoning_content")
        or message.get("thinking")
        or message.get("reasoning")
    )
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned empty content")
    if thinking is None and isinstance(content, str):
        match = re.match(r"\s*<think>(.*?)</think>(.*)\Z", content, flags=re.DOTALL | re.IGNORECASE)
        if match:
            thinking = match.group(1)
    return {
        "content": content,
        "thinking": thinking,
        "usage": _normalize_usage(data.get("usage")),
        "finish_reason": choice.get("finish_reason"),
    }

