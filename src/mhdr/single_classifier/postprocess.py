import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def evaluate_selection_threshold(
    model,
    df,
    dimensions,
    text_col="text",
    threshold=0.5,
):
    """
    Evaluate binary selection after thresholding.

    Returns:
        - micro precision
        - micro recall
        - micro f1
        - exact match accuracy
    """
    texts = df[text_col].astype(str).tolist()
    probs = model.predict_probs(texts)

    y_pred = (probs >= threshold).astype(np.int32)
    y_true = (df[dimensions].to_numpy(dtype=np.float32) > 0).astype(np.int32)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    exact_match = (y_pred == y_true).all(axis=1).mean()

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
    }


def search_best_threshold(
    model,
    val_df,
    dimensions,
    text_col="text",
    thresholds=None,
    sort_by="f1",
):
    """
    Search best threshold for binary selection.
    """
    if thresholds is None:
        thresholds = [
            0.10, 0.15, 0.20, 0.25, 0.30,
            0.35, 0.40, 0.45, 0.50, 0.55,
            0.60, 0.65, 0.70, 0.75, 0.80
        ]

    rows = []

    for th in thresholds:
        metrics = evaluate_selection_threshold(
            model=model,
            df=val_df,
            dimensions=dimensions,
            text_col=text_col,
            threshold=th,
        )
        rows.append(metrics)

    result = pd.DataFrame(rows)

    ascending = True if sort_by == "threshold" else False
    return result.sort_values(sort_by, ascending=ascending).reset_index(drop=True)