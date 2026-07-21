def precision_recall_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    
    return [precision, recall]