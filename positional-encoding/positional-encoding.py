import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(base, (dims // 2) * 2 / d_model)
    angles = positions * angle_rates
    pe = np.zeros((seq_len, d_model), dtype=float)
    pe[:, 0::2] = np.sin(angles[:, 0::2])   
    pe[:, 1::2] = np.cos(angles[:, 1::2])   
    
    return pe