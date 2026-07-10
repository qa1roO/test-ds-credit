"""Тесты классификатора документов по ключевым словам."""

import pytest

from classification import classify


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ДОГОВОР ПОСТАВКИ № 1\nПредмет договора", "contract"),
        ("СПЕЦИФИКАЦИЯ № 1\nНаименование товара", "spec"),
        ("СЧЁТ НА ОПЛАТУ № 2\nИтого к оплате", "invoice"),
        ("АКТ ВЫПОЛНЕННЫХ РАБОТ № 3", "act"),
        ("Универсальный передаточный документ (УПД)", "act"),
        ("случайный текст", "unknown"),
        ("", "unknown"),
    ],
)
def test_document_types(text: str, expected: str) -> None:
    label, confidence = classify(text)
    assert label == expected
    assert 0.0 <= confidence <= 1.0


def test_close_scores_return_unknown() -> None:
    label, _ = classify("ДОГОВОР ПОСТАВКИ. СПЕЦИФИКАЦИЯ")
    assert label == "unknown"


def test_confidence_uses_only_two_leaders() -> None:
    label, confidence = classify("Счет на оплату. Договор.")
    assert label == "invoice"
    assert confidence == pytest.approx(5 / 6, abs=0.001)


def test_confidence_is_one_when_second_score_is_zero() -> None:
    assert classify("Спецификация")[1] == 1.0


def test_required_invoice_example() -> None:
    doc_type, confidence = classify("Счёт на оплату №12 от 01.03.2025 ...")
    assert doc_type == "invoice"
    assert confidence > 0.5
