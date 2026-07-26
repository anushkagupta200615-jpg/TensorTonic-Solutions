import numpy as np
from collections import Counter
import math
from typing import List, Tuple

def tfidf_vectorizer(documents: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Tokenize documents
    tokenized_docs = [doc.lower().split() for doc in documents]
    
    # Build vocabulary (sorted alphabetically)
    vocab = sorted(set(term for doc in tokenized_docs for term in doc))
    vocab_index = {term: i for i, term in enumerate(vocab)}
    
    n_docs = len(documents)
    n_vocab = len(vocab)
    
    # Term Frequency (TF)
    tf = np.zeros((n_docs, n_vocab))
    for d_idx, doc in enumerate(tokenized_docs):
        if not doc:  # handle empty doc
            continue
        counts = Counter(doc)
        total_terms = len(doc)
        for term, count in counts.items():
            tf[d_idx, vocab_index[term]] = count / total_terms
    
    # Document Frequency (DF)
    df = np.zeros(n_vocab)
    for term, idx in vocab_index.items():
        df[idx] = sum(1 for doc in tokenized_docs if term in doc)
    
    # Inverse Document Frequency (IDF)
    idf = np.zeros(n_vocab)
    for idx, df_val in enumerate(df):
        if df_val > 0:
            idf[idx] = math.log(n_docs / df_val)
        else:
            idf[idx] = 0.0
    
    # TF-IDF = TF × IDF
    tfidf = tf * idf
    
    return tfidf, vocab
