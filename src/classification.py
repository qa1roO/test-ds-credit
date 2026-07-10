"""Классификация финансовых документов по ключевым словам."""

from __future__ import annotations

MIN_MARGIN = 1.0

KEYWORDS: dict[str, dict[str, float]] = {
    "contract": {
        "договор поставки": 4.0,
        "предмет договора": 3.0,
        "заключили настоящий договор": 3.0,
        "стороны": 1.0,
        "договор": 1.0,
    },
    "spec": {
        "спецификация": 5.0,
        "приложение к договору": 2.0,
        "наименование товара": 2.0,
    },
    "invoice": {
        "счет на оплату": 5.0,
        "счет №": 4.0,
        "к оплате": 2.0,
        "всего наименований": 2.0,
    },
    "act": {
        "акт выполненных работ": 5.0,
        "акт оказанных услуг": 5.0,
        "универсальный передаточный документ": 5.0,
        "упд": 4.0,
        "работы выполнены": 2.0,
    },
}


def classification_scores(text: str) -> dict[str, float]:
    """Рассчитать сумму весов ключевых признаков для каждого класса."""
    normalized = text.lower().replace("ё", "е")
    return {
        label: sum(
            weight for keyword, weight in keywords.items() if keyword in normalized
        )
        for label, keywords in KEYWORDS.items()
    }


def classify(text: str) -> tuple[str, float]:
    """Определить тип документа и вернуть эвристическую уверенность от 0 до 1."""
    if not text or not text.strip():
        return "unknown", 0.0

    scores = classification_scores(text)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    (best_label, best_score), (_, second_score) = ranked[:2]

    if best_score == 0:
        return "unknown", 0.0

    confidence = (
        1.0 if second_score == 0 else best_score / (best_score + second_score)
    )
    if best_score - second_score < MIN_MARGIN:
        return "unknown", round(confidence, 3)
    return best_label, round(confidence, 3)
