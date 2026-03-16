import numpy as np
import pandas as pd
from itertools import product


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def scores_to_ranked_vector(
    scores,
    dimensions,
    threshold=0.5,
):
    probs = sigmoid(scores)
    order = np.argsort(scores)[::-1]

    selected = [i for i in order if probs[i] >= threshold]

    out = {dim: 0 for dim in dimensions}

    for rank, idx in enumerate(selected, start=1):
        out[dimensions[idx]] = rank

    return out


def scores_to_ranked_vector_threshold_top1(
    scores,
    dimensions,
    threshold=0.5,
):
    """
    Threshold-based decoding, but always keep top-1 if nothing passes threshold.
    """
    probs = sigmoid(scores)
    order = np.argsort(scores)[::-1]

    selected = [i for i in order if probs[i] >= threshold]

    if len(selected) == 0 and len(order) > 0:
        selected = [order[0]]

    out = {dim: 0 for dim in dimensions}

    for rank, idx in enumerate(selected, start=1):
        out[dimensions[idx]] = rank

    return out


def weighted_l1(y_pred, y_true, miss_penalty=3.0):
    total = 0.0
    for yp, yt in zip(y_pred, y_true):
        if yt > 0 and yp == 0:
            total += miss_penalty
        else:
            total += abs(yp - yt)
    return total


def weighted_postprocess_l1(
    model,
    df,
    dimensions,
    text_col="text",
    threshold=0.5,
    miss_penalty=3.0,
):
    texts = df[text_col].astype(str).tolist()
    scores_all = model.predict_scores(texts)

    total = 0.0

    for scores, (_, row) in zip(scores_all, df.iterrows()):
        pred_dict = scores_to_ranked_vector_threshold_top1(
            scores=scores,
            dimensions=dimensions,
            threshold=threshold,
        )

        y_pred = np.array([pred_dict[d] for d in dimensions], dtype=np.float32)
        y_true = row[dimensions].to_numpy(dtype=np.float32)

        total += weighted_l1(
            y_pred=y_pred,
            y_true=y_true,
            miss_penalty=miss_penalty,
        )

    return total / len(df)