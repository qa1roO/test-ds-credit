"""Формирование компактных отчётов по документам, классам и предметам."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from tabulate import tabulate

from classification import classify
from extraction import FIELDS, extract
from llm_client import llm_available
from subject_check import check_subject

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"

EXPECTED_EXTRACTION: dict[str, dict[str, object | None]] = {
    "contract_001.txt": {
        "amount": 1_250_000.0,
        "date": "2025-03-01",
        "inn": "7701234567",
        "contractor": "ООО «ТехАгро»",
        "subject": "минеральные удобрения",
    },
    "spec_001.txt": {
        "amount": 1_250_000.0,
        "date": "2025-03-01",
        "inn": "7701234567",
        "contractor": "ООО «ТехАгро»",
        "subject": "Карбамид марки Б",
    },
    "invoice_001.txt": {
        "amount": 1_250_000.0,
        "date": "2025-03-03",
        "inn": "7701234567",
        "contractor": "ООО «ТехАгро»",
        "subject": "Карбамид марки Б",
    },
    "invoice_002.txt": {
        "amount": 900_000.0,
        "date": "2025-02-15",
        "inn": "5047123456",
        "contractor": "АО «АгроСнаб»",
        "subject": "поставка семян подсолнечника",
    },
    "act_001.txt": {
        "amount": 1_250_000.0,
        "date": "2025-03-24",
        "inn": "7701234567",
        "contractor": "ООО «ТехАгро»",
        "subject": "Карбамид марки Б",
    },
    "act_002.txt": {
        "amount": 500_000.0,
        "date": "2025-04-01",
        "inn": "504712345678",
        "contractor": "ИП Смирнов В.А.",
        "subject": "Внесение жидких комплексных удобрений",
    },
    "scan_ocr_001.txt": {
        "amount": 1_250_000.0,
        "date": "2025-03-01",
        "inn": "7701234567",
        "contractor": "ЗАО «ТехАгро»",
        "subject": "Карбамид марки Б",
    },
}

EXPECTED_CLASSIFICATION = {
    "contract_001.txt": "contract",
    "spec_001.txt": "spec",
    "invoice_001.txt": "invoice",
    "invoice_002.txt": "invoice",
    "act_001.txt": "act",
    "act_002.txt": "act",
    "scan_ocr_001.txt": "unknown",
}


def requirements_report() -> str:
    """Показать фактические результаты шести примеров из исходного ТЗ."""
    rows: list[list[object]] = []

    amount = extract("Сумма: 1 250 000,00 руб.")["amount"]
    rows.append(
        [
            "extract",
            "Сумма: 1 250 000,00 руб.",
            "amount=1250000.0",
            f"amount={amount}",
            "OK" if amount == 1_250_000.0 else "MISS",
        ]
    )

    inn = extract("ИНН 7701234567")["inn"]
    rows.append(
        [
            "extract",
            "ИНН 7701234567",
            "inn=7701234567",
            f"inn={inn}",
            "OK" if inn == "7701234567" else "MISS",
        ]
    )

    missing_amount = extract("без цифр")["amount"]
    rows.append(
        [
            "extract",
            "без цифр",
            "amount=None",
            f"amount={missing_amount}",
            "OK" if missing_amount is None else "MISS",
        ]
    )

    doc_type, class_confidence = classify("Счёт на оплату №12 от 01.03.2025 ...")
    class_ok = doc_type == "invoice" and class_confidence > 0.5
    rows.append(
        [
            "classify",
            "Счёт на оплату №12 от 01.03.2025 ...",
            "invoice, > 0.5",
            f"{doc_type}, {class_confidence:.3f}",
            "OK" if class_ok else "MISS",
        ]
    )

    for subject, expected in (
        ("Поставка минеральных удобрений", True),
        ("Аренда офисного помещения", False),
    ):
        matches, confidence, _reason = check_subject(subject)
        rows.append(
            [
                "check_subject",
                subject,
                str(expected),
                f"{matches}, {confidence:.2f}",
                "OK" if matches is expected else "MISS",
            ]
        )

    correct = sum(row[-1] == "OK" for row in rows)
    table = tabulate(
        [
            [function, _short(value, 45), expected, received, check]
            for function, value, expected, received, check in rows
        ],
        headers=["Функция", "Вход", "Ожидалось", "Получено", "Проверка"],
        tablefmt="github",
    )
    return f"{table}\n\nКорректно обязательных примеров: {correct}/{len(rows)}"


def _normalized(value: object | None) -> str:
    """Нормализовать текст для мягкого сравнения ожидаемого и полученного."""
    text = str(value or "").lower().replace("ё", "е")
    return " ".join(re.sub(r"[^a-zа-я0-9]+", " ", text, flags=re.IGNORECASE).split())


def field_matches(field: str, predicted: object | None, expected: object | None) -> bool:
    """Сравнить ожидаемое и извлечённое значение после простой нормализации."""
    if predicted is None or expected is None:
        return predicted is expected
    if field == "amount":
        return abs(float(predicted) - float(expected)) <= 0.01
    if field in {"date", "inn"}:
        return predicted == expected
    left, right = _normalized(predicted), _normalized(expected)
    return bool(left and (left in right or right in left))


def evaluate_documents() -> list[dict[str, Any]]:
    """Запустить публичную функцию extract для каждого документа датасета."""
    rows: list[dict[str, Any]] = []
    for filename, expected in EXPECTED_EXTRACTION.items():
        text = (DATASET_DIR / filename).read_text(encoding="utf-8")
        predicted = extract(text)
        checks = {
            field: field_matches(field, predicted.get(field), expected[field])
            for field in FIELDS
        }
        rows.append(
            {
                "filename": filename,
                "expected": expected,
                "predicted": predicted,
                "checks": checks,
            }
        )
    return rows


def _short(value: object | None, limit: int = 70) -> str:
    """Сократить длинное значение для компактной Markdown-таблицы."""
    text = "None" if value is None else " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_document_report(rows: list[dict[str, Any]]) -> str:
    """Сформировать сводную таблицу и отдельную таблицу несовпадений."""
    summary: list[list[object]] = []
    errors: list[list[object]] = []
    total_ok = 0
    for row in rows:
        checks = row["checks"]
        correct = sum(checks.values())
        total_ok += correct
        summary.append(
            [
                Path(row["filename"]).stem,
                *("✅" if checks[field] else "❌" for field in FIELDS),
                f"{correct}/{len(FIELDS)}",
            ]
        )
        for field in FIELDS:
            if not checks[field]:
                errors.append(
                    [
                        Path(row["filename"]).stem,
                        field,
                        _short(row["expected"][field]),
                        _short(row["predicted"].get(field)),
                    ]
                )

    summary_table = tabulate(
        summary,
        headers=["Документ", "Сумма", "Дата", "ИНН", "Контрагент", "Предмет", "Результат"],
        tablefmt="github",
    )
    error_table = (
        tabulate(
            errors,
            headers=["Документ", "Поле", "Ожидалось", "Получено"],
            tablefmt="github",
        )
        if errors
        else "Несовпадений нет."
    )
    total = len(rows) * len(FIELDS)
    accuracy = total_ok / total if total else 0.0
    return (
        f"### Сводка\n\n{summary_table}\n\n"
        f"Корректно полей: {total_ok}/{total} ({accuracy:.1%})\n\n"
        f"### Ошибки\n\n{error_table}"
    )


def document_report() -> str:
    """Запустить оценку извлечения и вернуть готовый Markdown-отчёт."""
    return format_document_report(evaluate_documents())


def classification_report() -> str:
    """Сформировать таблицу сравнения ожидаемого и полученного типа документа."""
    rows = []
    correct = 0
    for filename, expected in EXPECTED_CLASSIFICATION.items():
        text = (DATASET_DIR / filename).read_text(encoding="utf-8")
        predicted, confidence = classify(text)
        ok = predicted == expected
        correct += ok
        rows.append([Path(filename).stem, expected, predicted, confidence, "OK" if ok else "MISS"])
    table = tabulate(
        rows,
        headers=["Документ", "Эталон", "Результат", "Уверенность", "Проверка"],
        tablefmt="github",
    )
    return f"{table}\n\nКорректно документов: {correct}/{len(rows)}"


def load_subject_cases() -> list[tuple[str, str]]:
    """Загрузить примеры PASS, FAIL и EDGE из файла subjects_test.txt."""
    cases: list[tuple[str, str]] = []
    for raw_line in (DATASET_DIR / "subjects_test.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith(("PASS | ", "FAIL | ", "EDGE | ")):
            label, subject = line.split(" | ", 1)
            cases.append((label, subject))
    return cases


def subject_report() -> str:
    """Показать PASS, FAIL и EDGE вместе, не включая EDGE в accuracy."""
    rows: list[list[object]] = []
    strict_correct = 0
    strict_total = 0
    edge_total = 0

    for label, subject in load_subject_cases():
        matches, confidence, reason = check_subject(subject)
        decision = "PASS" if matches else "FAIL"
        if label == "EDGE":
            edge_total += 1
            rows.append(
                [
                    _short(subject, 60),
                    label,
                    decision,
                    confidence,
                    _short(reason, 70),
                    "—",
                ]
            )
            continue

        ok = decision == label
        strict_total += 1
        strict_correct += int(ok)
        rows.append(
            [
                _short(subject, 60),
                label,
                decision,
                confidence,
                _short(reason, 70),
                "OK" if ok else "MISS",
            ]
        )

    table = tabulate(
        rows,
        headers=[
            "Предмет",
            "Эталон",
            "Решение",
            "Уверенность",
            "Причина",
            "Проверка",
        ],
        tablefmt="github",
    )
    return (
        f"{table}\n\n"
        f"Корректно однозначных примеров: {strict_correct}/{strict_total}\n\n"
        f"Пограничных примеров: {edge_total}. Они не включаются в расчёт точности."
    )


def main(argv: list[str] | None = None) -> int:
    """Вывести один выбранный отчёт или все отчёты сразу."""
    parser = argparse.ArgumentParser(description="Отчёты по тестовому датасету")
    parser.add_argument(
        "--report",
        choices=("requirements", "documents", "classification", "subjects", "all"),
        default="all",
    )
    args = parser.parse_args(argv)

    if llm_available():
        print("Используется LLM\n")
    else:
        print("LLM недоступна или HF_TOKEN не задан используется обработка без LLM\n")

    reports = {
        "requirements": ("Обязательные примеры из ТЗ", requirements_report),
        "documents": ("Документы", document_report),
        "classification": ("Классификация", classification_report),
        "subjects": ("Предметы оплаты", subject_report),
    }
    selected = reports if args.report == "all" else {args.report: reports[args.report]}
    for title, report_factory in selected.values():
        print(f"## {title}\n\n{report_factory()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
