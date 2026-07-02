import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    if not seqs:
       return np.zeros((0,0),dtype=int)
    if max_len is None:
        max_len=max(len(s) for s in seqs)
    padded =[]
    for s in seqs:
        if len(s)>max_len:
            new_s=s[:max_len]
        else:
             new_s = s + [pad_value] * (max_len - len(s))
        padded.append(new_s)
    return np.array(padded, dtype=int)