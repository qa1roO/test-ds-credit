"""Проверка предмета оплаты на соответствие сельскохозяйственной программе."""

from __future__ import annotations

from llm_client import ask_llm, llm_available, parse_json_response

ALLOWED_KEYWORDS = {
    "удобрени",
    "карбамид",
    "кас-32",
    "гербицид",
    "фунгицид",
    "семена",
    "посевной материал",
    "трактор",
    "комбайн",
    "сельхозтехник",
    "запчаст",
    "дизельн",
    "гсм",
    "обработка почвы",
    "посев",
    "уборка урожая",
    "ремонт трактор",
    "ремонт комбайн",
    "ремонт сельхозтехник",
    "агроном",
    "страхование урожая",
}

DENIED_KEYWORDS = {
    "аренда офис",
    "офисное помещение",
    "офисная мебель",
    "канцеляр",
    "реклам",
    "seo",
    "разработка сайта",
    "корпоративного сайта",
    "юридическ",
    "клининг",
    "обучение",
}

AMBIGUOUS_KEYWORDS = (
    "аренд",
    "достав",
    "транспорт",
    "консультац",
    "консульт",
    "сопровожд",
)

SUBJECT_PROMPT = """Определи, соответствует ли предмет оплаты целям
сельскохозяйственного льготного кредита.

Разрешённые категории: удобрения, средства защиты растений, семена,
сельхозтехника, запчасти, ГСМ, полевые работы, ремонт сельхозтехники,
агрономические услуги.

Примеры нецелевых расходов: аренда офиса, офисная мебель, реклама, разработка
сайта, юридические услуги общего характера.

Верни только JSON:
{"matches": true, "confidence": 0.87, "reason": "Краткое объяснение"}

Предмет оплаты:
"""


def fallback_check_subject(subject: str) -> tuple[bool, float, str]:
    """Проверить очевидные разрешённые и запрещённые формулировки локально."""
    normalized = subject.lower().replace("ё", "е")
    if any(keyword in normalized for keyword in DENIED_KEYWORDS):
        return False, 0.9, "Предмет относится к нецелевым расходам"
    if any(keyword in normalized for keyword in ALLOWED_KEYWORDS):
        return (
            True,
            0.85,
            "Найдено соответствие разрешённой сельскохозяйственной категории",
        )
    return (
        False,
        0.5,
        "Не удалось уверенно отнести предмет к разрешённой категории",
    )


def _llm_subject_result(subject: str) -> tuple[bool, float, str] | None:
    """Получить решение LLM для неоднозначного предмета оплаты."""
    response = ask_llm(SUBJECT_PROMPT + subject, max_tokens=96)
    parsed = parse_json_response(response)
    if not parsed:
        return None

    matches = parsed.get("matches")
    confidence = parsed.get("confidence")
    reason = parsed.get("reason")
    if (
        not isinstance(matches, bool)
        or not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        return None

    normalized_confidence = min(1.0, max(0.0, float(confidence)))
    return matches, normalized_confidence, reason.strip()


def check_subject(subject: str) -> tuple[bool, float, str]:
    """Определить соответствие предмета программе, уверенность и причину решения."""
    local_result = fallback_check_subject(subject)
    normalized = subject.lower().replace("ё", "е")

    is_ambiguous = any(keyword in normalized for keyword in AMBIGUOUS_KEYWORDS)
    has_allowed = any(keyword in normalized for keyword in ALLOWED_KEYWORDS)
    has_denied = any(keyword in normalized for keyword in DENIED_KEYWORDS)
    obvious_denial = has_denied and not has_allowed

    if not is_ambiguous or obvious_denial or not llm_available():
        return local_result
    return _llm_subject_result(subject) or local_result
