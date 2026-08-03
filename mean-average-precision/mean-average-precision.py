import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) across queries.

    Parameters:
        y_true_list: list of arrays/lists of binary relevance {0,1}
        y_score_list: list of arrays/lists of real-valued scores
        k: optional cutoff rank (int). If None, use full length

    Returns:
        (mAP, ap_list): tuple of float and list of per-query AP values
    """
    ap_list = []

    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)

        # Sort by score descending
        order = np.argsort(-y_score)
        y_true = y_true[order]

        if k is not None:
            y_true_k = y_true[:k]
        else:
            y_true_k = y_true

        R = np.sum(y_true)  # total relevant items in the query
        if R == 0:
            ap_list.append(0.0)
            continue

        # Precision at each rank where item is relevant (within cutoff)
        rel_indices = np.where(y_true_k == 1)[0]
        precisions = [(i+1) / (idx+1) for i, idx in enumerate(rel_indices)]
        ap = np.sum(precisions) / R
        ap_list.append(ap)

    mAP = float(np.mean(ap_list)) if ap_list else 0.0
    return mAP, ap_list
