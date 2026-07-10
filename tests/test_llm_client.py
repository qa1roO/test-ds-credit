"""Тесты клиента Hugging Face без сетевых запросов."""

from types import SimpleNamespace

import huggingface_hub
import pytest

import llm_client


@pytest.fixture(autouse=True)
def reset_client_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Сбрасывать процессное состояние клиента между unit-тестами."""
    monkeypatch.setattr(llm_client, "_PROVIDER_DISABLED", False)
    monkeypatch.setattr(llm_client, "_LAST_REQUEST_STARTED_AT", None)


def _successful_response(content: str = "ok") -> SimpleNamespace:
    """Собрать минимальный объект успешного ответа клиента."""
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_json_parser_removes_thinking_and_service_text() -> None:
    response = '<think>reasoning</think>Ответ: {"amount": 100, "date": null}'
    assert llm_client.parse_json_response(response) == {
        "amount": 100,
        "date": None,
    }


def test_json_parser_returns_none_for_invalid_response() -> None:
    assert llm_client.parse_json_response("not json") is None


def test_max_tokens_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs

        def chat_completion(self, **kwargs: object) -> SimpleNamespace:
            captured["request_kwargs"] = kwargs
            return _successful_response()

    monkeypatch.setenv("HF_TOKEN", "hf_test_secret")
    monkeypatch.setattr(huggingface_hub, "InferenceClient", FakeClient)

    assert llm_client.ask_llm("prompt", max_tokens=73) == "ok"
    assert captured["request_kwargs"]["max_tokens"] == 73  
    assert captured["client_kwargs"]["provider"] == "auto" 


def test_first_request_does_not_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat_completion(self, **_kwargs: object) -> SimpleNamespace:
            return _successful_response()

    monkeypatch.setenv("HF_TOKEN", "hf_test_secret")
    monkeypatch.setattr(huggingface_hub, "InferenceClient", FakeClient)
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        llm_client.time,
        "sleep",
        lambda _seconds: pytest.fail("первый запрос не должен ждать"),
    )

    assert llm_client.ask_llm("first") == "ok"


def test_second_request_waits_for_remaining_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monotonic_values = iter((0.0, 0.0, 4.0, 10.0))

    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat_completion(self, **_kwargs: object) -> SimpleNamespace:
            return _successful_response()

    monkeypatch.setenv("HF_TOKEN", "hf_test_secret")
    monkeypatch.setattr(huggingface_hub, "InferenceClient", FakeClient)
    monkeypatch.setattr(llm_client, "LLM_REQUEST_INTERVAL_SECONDS", 10.0)
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)

    assert llm_client.ask_llm("first") == "ok"
    assert llm_client.ask_llm("second") == "ok"
    assert sleeps == [pytest.approx(6.0)]


def test_disabled_provider_does_not_wait_or_create_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_client, "_PROVIDER_DISABLED", True)
    monkeypatch.setattr(
        huggingface_hub,
        "InferenceClient",
        lambda **_kwargs: pytest.fail("клиент не должен создаваться"),
    )
    monkeypatch.setattr(
        llm_client.time,
        "sleep",
        lambda _seconds: pytest.fail("отключённый provider не должен ждать"),
    )

    assert llm_client.ask_llm("prompt") is None


@pytest.mark.parametrize("status", [401, 402, 403])
def test_auth_or_billing_error_disables_following_requests_and_redacts_log(
    status: int,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = 0
    secret = "hf_test_secret"

    class ProviderError(Exception):
        response = SimpleNamespace(
            status_code=status,
            text=f"Credits exhausted for {secret}. " + "x" * 500,
        )

    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal calls
            calls += 1

        def chat_completion(self, **_kwargs: object) -> None:
            raise ProviderError()

    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setattr(huggingface_hub, "InferenceClient", FakeClient)
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(
        llm_client.time,
        "sleep",
        lambda _seconds: pytest.fail("после блокирующей ошибки ожидания быть не должно"),
    )

    with caplog.at_level("WARNING"):
        assert llm_client.ask_llm("first") is None
        assert llm_client.ask_llm("second") is None

    assert calls == 1
    assert llm_client._PROVIDER_DISABLED is True
    assert len(caplog.records) == 1
    assert f"status={status}" in caplog.text
    assert "Credits exhausted" in caplog.text
    assert secret not in caplog.text
    assert "[REDACTED]" in caplog.text
    assert len(caplog.records[0].getMessage()) < 450


def test_provider_error_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat_completion(self, **_kwargs: object) -> None:
            raise RuntimeError("temporary failure")

    monkeypatch.setenv("HF_TOKEN", "hf_test_secret")
    monkeypatch.setattr(huggingface_hub, "InferenceClient", FakeClient)

    assert llm_client.ask_llm("prompt") is None
