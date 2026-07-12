"""Тесты извлечения, нормализации и подменённого ответа LLM."""

import pytest

import extraction


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Отключить LLM только для тестов локального fallback."""
    monkeypatch.setattr(extraction, "llm_available", lambda: False)


def test_required_amount_example(offline: None) -> None:
    assert extraction.extract("Сумма: 1 250 000,00 руб.")["amount"] == 1_250_000.0


def test_required_inn_example(offline: None) -> None:
    assert extraction.extract("ИНН 7701234567")["inn"] == "7701234567"


def test_required_missing_amount_example(offline: None) -> None:
    assert extraction.extract("без цифр")["amount"] is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Общая стоимость услуг по договору составляет 420 000 рублей", 420_000.0),
        ("Итого без НДС: 100 000 руб.\nИтого с НДС: 120 000 руб.", 120_000.0),
        ("Итого стоимость работ: 75 500,50 руб.", 75_500.5),
        ("| Наименование | Сумма |\nИтого: 12 300 руб.", 12_300.0),
    ],
)
def test_fallback_recognizes_common_total_labels(
    offline: None,
    text: str,
    expected: float,
) -> None:
    assert extraction.extract(text)["amount"] == expected


def test_fallback_handles_common_ocr_digit_substitutions(offline: None) -> None:
    result = extraction.extract(
        "Cyммa: l 25O OOO pyб\nДата: O1.O3.2O25\nИНН 77O1234567"
    )

    assert result["amount"] == 1_250_000.0
    assert result["date"] == "2025-03-01"
    assert result["inn"] == "7701234567"


def test_fallback_extracts_subject_from_first_table_row(offline: None) -> None:
    text = """| № | Товар / услуга                    | Количество |
| 1 | Техническое обслуживание оборудования | 1 |"""
    assert extraction.extract(text)["subject"] == "Техническое обслуживание оборудования"


def test_fallback_extracts_subject_from_contract_clause(offline: None) -> None:
    text = (
        "Поставщик обязуется передать в собственность Покупателя запасные части "
        "для трактора, а Покупатель обязуется принять товар."
    )
    assert extraction.extract(text)["subject"] == "запасные части для трактора"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1 250 000,00 руб.", 1_250_000.0),
        ("1250000.00 ₽", 1_250_000.0),
        ("1,250,000.00 RUB", 1_250_000.0),
        ("1.250.000,00 руб.", 1_250_000.0),
        ("1 250 000 руб. 00 коп.", 1_250_000.0),
    ],
)
def test_amount_normalization(raw: str, expected: float) -> None:
    assert extraction.normalize_amount(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("01.03.2025", "2025-03-01"),
        ("1 марта 2025 г.", "2025-03-01"),
        ("03/01/25", "2025-01-03"),
        ("2025-03-01", "2025-03-01"),
        ("31.02.2025", None),
    ],
)
def test_date_normalization(raw: str, expected: str | None) -> None:
    assert extraction.normalize_date(raw) == expected


def test_mocked_llm_result_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(extraction, "llm_available", lambda: True)

    def mocked_llm(prompt: str, *, max_tokens: int) -> str:
        captured["prompt"] = prompt
        captured["max_tokens"] = max_tokens
        return """{
            "amount": "1 250 000,00 руб.",
            "date": "01.03.2025",
            "inn": "7701234567",
            "contractor": "ООО ТехАгро",
            "subject": "Карбамид"
        }"""

    monkeypatch.setattr(extraction, "ask_llm", mocked_llm)
    result = extraction.extract("любой неизвестный формат")

    assert result == {
        "amount": 1_250_000.0,
        "date": "2025-03-01",
        "inn": "7701234567",
        "contractor": "ООО ТехАгро",
        "subject": "Карбамид",
    }
    assert captured["max_tokens"] == 250
    assert "не обобщай и не перефразируй" in str(captured["prompt"])


def test_llm_error_uses_whole_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extraction, "llm_available", lambda: True)
    monkeypatch.setattr(extraction, "ask_llm", lambda _prompt, **_kwargs: None)
    result = extraction.extract("Сумма: 500 000 руб. ИНН 7701234567")
    assert result["amount"] == 500_000.0
    assert result["inn"] == "7701234567"


def test_empty_text_does_not_check_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        extraction,
        "llm_available",
        lambda: (_ for _ in ()).throw(AssertionError("LLM не должна вызываться")),
    )
    assert extraction.extract("   ") == extraction.EMPTY_RESULT
