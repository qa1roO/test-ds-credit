"""Тесты LLM-разбора и локальной проверки предмета оплаты."""

import pytest

import subject_check


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Отключить LLM для проверки локального fallback."""
    monkeypatch.setattr(subject_check, "llm_available", lambda: False)


def test_allowed_subject_fallback(offline: None) -> None:
    matches, confidence, reason = subject_check.check_subject("Поставка карбамида")
    assert matches is True
    assert confidence > 0.5
    assert reason


def test_denied_subject_fallback(offline: None) -> None:
    matches, confidence, reason = subject_check.check_subject(
        "Аренда офисного помещения"
    )
    assert matches is False
    assert confidence > 0.5
    assert reason


def test_unknown_subject_fallback(offline: None) -> None:
    assert subject_check.check_subject("Прочие услуги")[0:2] == (False, 0.5)


def test_denied_keyword_has_priority(offline: None) -> None:
    assert subject_check.check_subject("Обучение работе с сельхозтехникой")[0] is False


def test_mocked_llm_subject_result(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(subject_check, "llm_available", lambda: True)

    def mocked_llm(_prompt: str, *, max_tokens: int) -> str:
        captured["max_tokens"] = max_tokens
        return (
            '{"matches": true, "confidence": 0.87, '
            '"reason": "Разрешённые семена"}'
        )

    monkeypatch.setattr(subject_check, "ask_llm", mocked_llm)
    assert subject_check.check_subject("Консультация агронома") == (
        True,
        0.87,
        "Разрешённые семена",
    )
    assert captured["max_tokens"] == 96


def test_invalid_llm_subject_result_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject_check, "llm_available", lambda: True)
    monkeypatch.setattr(
        subject_check,
        "ask_llm",
        lambda _prompt, **_kwargs: "invalid",
    )
    result = subject_check.check_subject(
        "Транспортные услуги по доставке удобрений"
    )
    assert result[0] is True
    assert result[1] == 0.85


def test_obvious_pass_does_not_call_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject_check, "llm_available", lambda: True)
    monkeypatch.setattr(
        subject_check,
        "_llm_subject_result",
        lambda _subject: (_ for _ in ()).throw(AssertionError("лишний вызов LLM")),
    )
    assert subject_check.check_subject("Поставка карбамида")[0] is True


def test_obvious_fail_does_not_call_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject_check, "llm_available", lambda: True)
    monkeypatch.setattr(
        subject_check,
        "_llm_subject_result",
        lambda _subject: (_ for _ in ()).throw(AssertionError("лишний вызов LLM")),
    )
    assert subject_check.check_subject("Аренда офисного помещения")[0] is False


@pytest.mark.parametrize(
    "subject",
    [
        "Транспортные услуги по доставке удобрений до склада",
        "Аренда сельскохозяйственной техники на период уборки урожая",
        "Услуги агронома-консультанта по подбору схемы удобрений",
    ],
)
def test_ambiguous_subject_calls_llm(
    subject: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    monkeypatch.setattr(subject_check, "llm_available", lambda: True)

    def mocked_result(_subject: str) -> tuple[bool, float, str]:
        nonlocal calls
        calls += 1
        return True, 0.7, "Пограничный случай"

    monkeypatch.setattr(subject_check, "_llm_subject_result", mocked_result)
    subject_check.check_subject(subject)
    assert calls == 1


def test_generic_repair_is_not_allowed(offline: None) -> None:
    assert subject_check.check_subject("Ремонт офисного помещения")[0] is False


def test_combine_repair_stays_allowed(offline: None) -> None:
    assert subject_check.check_subject("Ремонт комбайна")[0] is True
