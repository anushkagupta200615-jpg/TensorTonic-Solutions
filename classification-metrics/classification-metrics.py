import numpy as np
from collections import Counter

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, and F1 for single-label classification.
    Supports averaging: 'micro' | 'macro' | 'weighted' | 'binary'.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n = len(y_true)

    # Accuracy
    accuracy = np.mean(y_true == y_pred)

    # Per-class counts
    counts = {lbl: {"TP":0,"FP":0,"FN":0} for lbl in labels}
    support = Counter(y_true)

    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            counts[yt]["TP"] += 1
        else:
            counts[yp]["FP"] += 1
            counts[yt]["FN"] += 1

    def safe_div(num, den): return num/den if den>0 else 0.0

    # Per-class metrics
    per_class = {}
    for lbl in labels:
        TP, FP, FN = counts[lbl]["TP"], counts[lbl]["FP"], counts[lbl]["FN"]
        prec = safe_div(TP, TP+FP)
        rec  = safe_div(TP, TP+FN)
        f1   = safe_div(2*prec*rec, prec+rec)
        per_class[lbl] = (prec, rec, f1)

    # Averaging strategies
    if average == "micro":
        TP = sum(c["TP"] for c in counts.values())
        FP = sum(c["FP"] for c in counts.values())
        FN = sum(c["FN"] for c in counts.values())
        prec = safe_div(TP, TP+FP)
        rec  = safe_div(TP, TP+FN)
        f1   = safe_div(2*prec*rec, prec+rec)

    elif average == "macro":
        prec = np.mean([p[0] for p in per_class.values()])
        rec  = np.mean([p[1] for p in per_class.values()])
        f1   = np.mean([p[2] for p in per_class.values()])

    elif average == "weighted":
        total = sum(support.values())
        prec = sum(per_class[lbl][0]*support[lbl] for lbl in labels)/total
        rec  = sum(per_class[lbl][1]*support[lbl] for lbl in labels)/total
        f1   = sum(per_class[lbl][2]*support[lbl] for lbl in labels)/total

    elif average == "binary":
        TP, FP, FN = counts[pos_label]["TP"], counts[pos_label]["FP"], counts[pos_label]["FN"]
        prec = safe_div(TP, TP+FP)
        rec  = safe_div(TP, TP+FN)
        f1   = safe_div(2*prec*rec, prec+rec)

    else:
        raise ValueError("average must be 'micro', 'macro', 'weighted', or 'binary'")

    return {
        "accuracy": float(accuracy),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }
