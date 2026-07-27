import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Compute BM25 scores for each document given a tokenized query.
    Returns np.ndarray of shape (len(docs),) with float scores.
    """
    N = len(docs)
    if N == 0:
        return np.array([], dtype=float)

    # Document lengths and average length
    doc_lengths = np.array([len(doc) for doc in docs], dtype=float)
    avgdl = doc_lengths.mean() if N > 0 else 0.0

    # Document frequency for each query term
    df = {}
    for term in set(query_tokens):
        df[term] = sum(1 for doc in docs if term in doc)

    # IDF values
    idf = {}
    for term in query_tokens:
        df_t = df.get(term, 0)
        if df_t == 0:
            idf[term] = 0.0
        else:
            idf[term] = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)

    # BM25 scoring
    scores = np.zeros(N, dtype=float)
    for d_idx, doc in enumerate(docs):
        if not doc:
            continue
        counts = Counter(doc)
        dl = doc_lengths[d_idx]
        norm = (1 - b) + b * (dl / avgdl)
        for term in query_tokens:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * norm
            scores[d_idx] += idf[term] * (numerator / denominator)

    return scores
