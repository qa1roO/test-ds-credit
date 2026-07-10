"""Минимальный клиент Hugging Face для LLM-функций проекта."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_PROVIDER = "auto"
LLM_REQUEST_INTERVAL_SECONDS = 10.0
_PROVIDER_DISABLED = False
_LAST_REQUEST_STARTED_AT: float | None = None

load_dotenv(PROJECT_ROOT / ".env")


def llm_available() -> bool:
    """Проверить наличие токена и доступность провайдера в текущем процессе."""
    token_exists = bool((os.getenv("HF_TOKEN") or "").strip())
    return token_exists and not _PROVIDER_DISABLED


def parse_json_response(response: str | None) -> dict[str, Any] | None:
    """Извлечь первый JSON-объект из ответа модели и удалить блоки thinking."""
    if not response:
        return None

    cleaned = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            parsed, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _wait_before_request() -> None:
    """Выдержать минимальный интервал между реальными запросами к LLM."""
    global _LAST_REQUEST_STARTED_AT

    now = time.monotonic()
    if _LAST_REQUEST_STARTED_AT is not None:
        elapsed = now - _LAST_REQUEST_STARTED_AT
        remaining = LLM_REQUEST_INTERVAL_SECONDS - elapsed
        if remaining > 0:
            LOGGER.debug("Ожидание %.2f с перед запросом к LLM", remaining)
            time.sleep(remaining)
    _LAST_REQUEST_STARTED_AT = time.monotonic()


def ask_llm(prompt: str, *, max_tokens: int = 200) -> str | None:
    """Отправить запрос в Hugging Face или вернуть None при ошибке провайдера."""
    global _PROVIDER_DISABLED

    if _PROVIDER_DISABLED:
        return None

    token = (os.getenv("HF_TOKEN") or "").strip()
    if not token:
        return None

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(
            model=os.getenv("HF_MODEL_ID") or DEFAULT_MODEL_ID,
            provider=os.getenv("HF_PROVIDER") or DEFAULT_PROVIDER,
            token=token,
            timeout=30,
        )
        _wait_before_request()
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return str(content) if content is not None else None
    except Exception as error:  # SDK провайдеров не имеют общего типа ошибки.
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        detail = " ".join(str(getattr(response, "text", "")).split())
        detail = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", detail)[:300]
        LOGGER.warning(
            "Hugging Face недоступен, используется fallback: status=%s, detail=%s",
            status,
            detail or type(error).__name__,
        )
        if status in {401, 402, 403}:
            _PROVIDER_DISABLED = True
        return None
