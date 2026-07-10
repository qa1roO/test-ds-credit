"""Тесты форматирования отчётов без обращения к внешним сервисам."""

import pytest

import evaluation


def test_requirements_report_has_six_successful_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mocked_extract(text: str) -> dict[str, object | None]:
        if text.startswith("Сумма"):
            amount, inn = 1_250_000.0, None
        elif text.startswith("ИНН"):
            amount, inn = None, "7701234567"
        else:
            amount, inn = None, None
        return {
            "amount": amount,
            "date": None,
            "inn": inn,
            "contractor": None,
            "subject": None,
        }

    monkeypatch.setattr(evaluation, "extract", mocked_extract)
    monkeypatch.setattr(evaluation, "classify", lambda _text: ("invoice", 1.0))
    monkeypatch.setattr(
        evaluation,
        "check_subject",
        lambda subject: ("удобрений" in subject, 0.9, "mock"),
    )

    report = evaluation.requirements_report()

    assert "Корректно обязательных примеров: 6/6" in report
    assert report.count("| extract") == 3
    assert report.count("| classify") == 1
    assert report.count("| check_subject") == 2


def test_all_report_sections_have_required_order(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(evaluation, "llm_available", lambda: False)
    monkeypatch.setattr(evaluation, "requirements_report", lambda: "requirements")
    monkeypatch.setattr(evaluation, "document_report", lambda: "documents")
    monkeypatch.setattr(evaluation, "classification_report", lambda: "classes")
    monkeypatch.setattr(evaluation, "subject_report", lambda: "subjects")

    assert evaluation.main(["--report", "all"]) == 0
    output = capsys.readouterr().out
    headings = (
        "## Обязательные примеры из ТЗ",
        "## Документы",
        "## Классификация",
        "## Предметы оплаты",
    )

    positions = [output.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_main_reports_llm_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(evaluation, "llm_available", lambda: True)
    monkeypatch.setattr(evaluation, "requirements_report", lambda: "requirements")

    assert evaluation.main(["--report", "requirements"]) == 0


def test_document_table_format_is_preserved() -> None:
    values = {
        "amount": 100.0,
        "date": "2025-01-01",
        "inn": "7701234567",
        "contractor": "ООО Тест",
        "subject": "Семена",
    }
    report = evaluation.format_document_report(
        [
            {
                "filename": "invoice_test.txt",
                "expected": values,
                "predicted": values,
                "checks": {field: True for field in values},
            }
        ]
    )
    assert "| Документ" in report
    assert "| Сумма" in report
    assert "| Контрагент" in report
    assert "| Предмет" in report
    assert "5/5" in report
    assert "Корректно полей: 5/5 (100.0%)" in report


def test_subject_report_combines_edge_without_adding_it_to_accuracy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = [
        ("PASS", "Семена"),
        ("FAIL", "Офис"),
        ("EDGE", "Доставка удобрений"),
    ]
    results = {
        "Семена": (True, 0.9, "Разрешено"),
        "Офис": (False, 0.9, "Не разрешено"),
        "Доставка удобрений": (True, 0.7, "Пограничный случай"),
    }
    monkeypatch.setattr(evaluation, "load_subject_cases", lambda: cases)
    monkeypatch.setattr(
        evaluation,
        "check_subject",
        lambda subject: results[subject],
    )

    report = evaluation.subject_report()
    assert "Корректно однозначных примеров: 2/2" in report
    assert (
        "Пограничных примеров: 1. Они не включаются в расчёт точности."
        in report
    )
    assert "| EDGE" in report
    edge_row = next(line for line in report.splitlines() if "Доставка удобрений" in line)
    assert "Пограничный случай" in edge_row
    assert "—" in edge_row
    assert "| OK" not in edge_row
