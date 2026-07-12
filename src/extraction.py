"""Извлечение реквизитов документа через LLM с fallback если возникли проблемы с LLM."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from llm_client import ask_llm, llm_available, parse_json_response

FIELDS = ("amount", "date", "inn", "contractor", "subject")
EMPTY_RESULT: dict[str, object | None] = {field: None for field in FIELDS}

EXTRACTION_PROMPT = """Извлеки реквизиты из финансового документа.

Верни только JSON:
{
  "amount": number|null,
  "date": "YYYY-MM-DD"|null,
  "inn": string|null,
  "contractor": string|null,
  "subject": string|null
}

Правила:
- amount — итоговая сумма документа или сумма к оплате.
- Не выбирай НДС, цену за единицу, количество или номер документа.
- date — дата самого документа, а не срок оплаты или поставки.
- contractor — поставщик, исполнитель, продавец или подрядчик.
- Не выбирай покупателя, заказчика или банк.
- inn должен относиться к contractor.
- subject — товар, работа, услуга или предмет договора.
- subject верни краткой цитатой из документа, не обобщай и не перефразируй.
- Если поле отсутствует, верни null.

Документ:
"""

MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}

OCR_DIGIT_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "О": "0",
        "о": "0",
        "I": "1",
        "l": "1",
        "|": "1",
    }
)
OCR_DIGIT = r"[0-9OОoоIl]"


def clean_text(value: object | None) -> str | None:
    """Убирает лишние пробелы и разделители из строкового значения."""
    if value is None:
        return None
    cleaned = " ".join(str(value).split()).strip(" ,;:|")
    return cleaned or None


def _plain_number(value: str) -> float | None:
    """Преобразовывает строку с разделителями тысяч и дробной частью в число."""
    normalized = value.translate(OCR_DIGIT_TRANSLATION).replace("\u00a0", " ")
    number = re.sub(r"[^\d,.]", "", normalized)
    if not number:
        return None

    if "," in number and "." in number:
        decimal = "," if number.rfind(",") > number.rfind(".") else "."
        number = number.replace("." if decimal == "," else ",", "")
        number = number.replace(decimal, ".")
    elif "," in number:
        parts = number.split(",")
        number = (
            "".join(parts)
            if len(parts) > 2 and all(len(part) == 3 for part in parts[1:])
            else "".join(parts[:-1]) + "." + parts[-1]
        )
    elif number.count(".") > 1:
        parts = number.split(".")
        number = (
            "".join(parts)
            if all(len(part) == 3 for part in parts[1:])
            else "".join(parts[:-1]) + "." + parts[-1]
        )

    try:
        return float(number)
    except ValueError:
        return None


def normalize_amount(value: object | None) -> float | None:
    """Нормализовывает основные российские и международные форматы суммы."""
    if value is None:
        return None

    text = str(value)
    rubles_and_kopecks = re.search(
        r"(?P<rub>\d[\d\s.,]*)\s*(?:руб\.?|рубля|рублей|₽|RUB)\s*"
        r"(?P<kop>\d{1,2})\s*коп",
        text,
        re.IGNORECASE,
    )
    if rubles_and_kopecks:
        rubles = _plain_number(rubles_and_kopecks.group("rub"))
        if rubles is not None:
            return rubles + int(rubles_and_kopecks.group("kop")) / 100

    text_without_currency = re.sub(
        r"(?i)руб(?:\.|ля|лей)?|RUB|₽", "", text
    )
    return _plain_number(text_without_currency)


def _valid_date(day: int, month: int, year: int) -> str | None:
    """Проверяет календарную дату и возврашает её в формате ISO."""
    year = year + 2000 if year < 100 else year
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def normalize_date(value: object | None) -> str | None:
    """Нормализовывает поддерживаемые форматы даты в строку YYYY-MM-DD."""
    if value is None:
        return None

    text = str(value).translate(OCR_DIGIT_TRANSLATION).lower().replace("ё", "е")
    iso = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if iso:
        year, month, day = (int(part) for part in iso.groups())
        return _valid_date(day, month, year)

    numeric = re.search(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b", text)
    if numeric:
        return _valid_date(*(int(part) for part in numeric.groups()))

    textual = re.search(
        rf"\b(\d{{1,2}})\s+({'|'.join(MONTHS)})\s+(\d{{2,4}})\b",
        text,
    )
    if textual:
        return _valid_date(
            int(textual.group(1)),
            MONTHS[textual.group(2)],
            int(textual.group(3)),
        )
    return None


def normalize_inn(value: object | None) -> str | None:
    """Вернуть ИНН из 10/12 цифр с учётом частых OCR-подмен."""
    normalized = str(value or "").translate(OCR_DIGIT_TRANSLATION)
    digits = re.sub(r"\D", "", normalized)
    return digits if len(digits) in (10, 12) else None


def normalize_result(data: dict[str, Any]) -> dict[str, object | None]:
    """Привести словарь от LLM к публичному формату функции extract."""
    return {
        "amount": normalize_amount(data.get("amount")),
        "date": normalize_date(data.get("date")),
        "inn": normalize_inn(data.get("inn")),
        "contractor": clean_text(data.get("contractor")),
        "subject": clean_text(data.get("subject")),
    }


def llm_extract(text: str) -> dict[str, object | None] | None:
    """Получить поля документа от LLM и нормализовать ответ."""
    response = ask_llm(EXTRACTION_PROMPT + text, max_tokens=250)
    parsed = parse_json_response(response)
    return normalize_result(parsed) if parsed is not None else None


MONEY_PATTERN = (
    rf"(?:{OCR_DIGIT}{{1,3}}(?:[ \u00a0.,]{OCR_DIGIT}{{3}})+|{OCR_DIGIT}+)"
    rf"(?:[,.]{OCR_DIGIT}{{1,2}})?\s*(?:руб\.?|рублей|₽|RUB)?"
    rf"(?:\s*{OCR_DIGIT}{{1,2}}\s*коп\.?)?"
)
DATE_PATTERN = (
    rf"{OCR_DIGIT}{{1,2}}[./-]{OCR_DIGIT}{{1,2}}[./-]{OCR_DIGIT}{{2,4}}|"
    rf"{OCR_DIGIT}{{1,2}}\s+(?:{'|'.join(MONTHS)})\s+"
    rf"{OCR_DIGIT}{{2,4}}\s*(?:г\.?)?"
)

AMOUNT_PATTERNS = (
    rf"(?:сумма(?:\s+к\s+оплате)?|[cс][yу]мм[aа]|к\s+оплате|"
    rf"итого\s+(?:к\s+оплате|с\s+ндс|стоимость\s+(?:работ|товаров?|услуг)))"
    rf"\s*:?[ \t]*({MONEY_PATTERN})",
    rf"общая\s+стоимость[^\n.:]{{0,100}}?\s+составляет\s*:?[ \t\n]*"
    rf"({MONEY_PATTERN})",
    rf"стоимость\s+(?:работ|товаров?|услуг)[^\n.:]{{0,60}}?\s+составляет"
    rf"\s*:?[ \t\n]*({MONEY_PATTERN})",
    rf"итого(?!\s+без\s+ндс)\s*:?[ \t]*({MONEY_PATTERN})",
    rf"всего[^\n]{{0,50}}?\bна\s+сумму\s*:?[ \t]*({MONEY_PATTERN})",
)


def _extract_subject(text: str) -> str | None:
    """Найти предмет в подписи, первой строке таблицы или типовой формулировке."""
    labeled = re.search(
        r"(?im)^\s*(?:предмет(?:\s+оплаты)?|назначение\s+платежа|"
        r"наименование(?:\s+(?:товара|работы|услуги))?)\s*:\s*(.+)$",
        text,
    )
    if labeled:
        return clean_text(labeled.group(1))

    table_row = re.search(r"(?m)^\|\s*1\s*\|\s*([^|\n]+?)\s*\|", text)
    if table_row:
        return clean_text(table_row.group(1))

    list_intro = re.search(
        r"(?i)следующ\w*\s+(?:работ\w*|товар\w*|услуг\w*)\s*:", text
    )
    if list_intro:
        first_item = re.search(
            r"(?m)^\s*1[.)]\s+(.+?)(?=\s+[—–-]\s*|$)",
            text[list_intro.end() :],
        )
        if first_item:
            return clean_text(first_item.group(1))

    contractual = re.search(
        r"(?is)\b(?:поставщик|продавец|исполнитель|подрядчик)\s+обязуется\s+"
        r"(?:передать|поставить|выполнить|оказать)"
        r"(?:\s+в\s+собственность\s+\S+)?\s+(.+?)"
        r"(?=\s*\(|,|;|\s+а\s+(?:покупатель|заказчик)\b|\n\s*\n)",
        text,
    )
    return clean_text(contractual.group(1)) if contractual else None


def fallback_extract(text: str) -> dict[str, object | None]:
    """Извлечь только явно подписанные поля без обращения к внешнему API."""
    if not text or not text.strip():
        return EMPTY_RESULT.copy()

    amount_match = next(
        (
            match
            for pattern in AMOUNT_PATTERNS
            if (match := re.search(pattern, text, re.IGNORECASE))
        ),
        None,
    )
    inn_match = re.search(
        rf"(?i)\bИНН(?:/КПП)?\s*:?[ \t]*({OCR_DIGIT}{{10}}|{OCR_DIGIT}{{12}})\b",
        text,
    )
    date_match = re.search(DATE_PATTERN, text, re.IGNORECASE)
    contractor_match = re.search(
        r"(?im)(?:поставщик|исполнитель|продавец|подрядчик)\s*:\s*"
        r"((?:ООО|АО|ПАО|ЗАО|ИП)\s+(?:[«\"].+?[»\"]|[^,\n;]+))",
        text,
    )
    subject = _extract_subject(text)

    return {
        "amount": normalize_amount(amount_match.group(1)) if amount_match else None,
        "date": normalize_date(date_match.group()) if date_match else None,
        "inn": normalize_inn(inn_match.group(1)) if inn_match else None,
        "contractor": clean_text(contractor_match.group(1))
        if contractor_match
        else None,
        "subject": subject,
    }


def extract(text: str) -> dict[str, object | None]:
    """Извлечь сумму, дату, ИНН, контрагента и предмет из текста документа."""
    if not text or not text.strip():
        return EMPTY_RESULT.copy()

    if llm_available():
        llm_result = llm_extract(text)
        if llm_result is not None:
            return llm_result
    return fallback_extract(text)
