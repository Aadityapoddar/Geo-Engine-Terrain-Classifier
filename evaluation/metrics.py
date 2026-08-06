"""Deterministic metrics derived from a fixed-order confusion matrix."""


def _safe_divide(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics_from_matrix(matrix, labels):
    """Return OA, kappa, macro F1, and per-class metrics.

    Missing precision/recall/F1 values are represented as ``None`` so absent
    classes cannot silently receive a perfect or zero score.
    """
    labels = list(labels)
    matrix = [list(row) for row in matrix]
    size = len(labels)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("Confusion matrix must be square and match label count")
    if any(value < 0 for row in matrix for value in row):
        raise ValueError("Confusion matrix counts cannot be negative")

    row_totals = [sum(row) for row in matrix]
    column_totals = [sum(matrix[row][column] for row in range(size))
                     for column in range(size)]
    total = sum(row_totals)
    correct = sum(matrix[index][index] for index in range(size))
    overall_accuracy = _safe_divide(correct, total)
    if total:
        expected = sum(row_totals[i] * column_totals[i] for i in range(size)) / total**2
        kappa = _safe_divide(overall_accuracy - expected, 1 - expected)
    else:
        kappa = None

    per_class = {}
    precision_values = []
    recall_values = []
    f1_values = []
    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        precision = _safe_divide(true_positive, column_totals[index])
        recall = _safe_divide(true_positive, row_totals[index])
        if precision is not None:
            precision_values.append(precision)
        if recall is not None:
            recall_values.append(recall)
        if precision is None or recall is None or precision + recall == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)
            f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": row_totals[index],
            "predicted": column_totals[index],
        }

    return {
        "confusion_matrix": matrix,
        "labels": labels,
        "sample_count": total,
        "overall_accuracy": overall_accuracy,
        "kappa": kappa,
        "macro_precision": (
            sum(precision_values) / len(precision_values) if precision_values else None
        ),
        "macro_recall": (
            sum(recall_values) / len(recall_values) if recall_values else None
        ),
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "per_class": per_class,
    }
