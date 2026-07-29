from collections import Counter
import numpy as np

def majority_classifier(y_train, X_test):
    freq = Counter(y_train)
    majority_class = max(freq, key=freq.get)
    return [majority_class] * len(X_test)