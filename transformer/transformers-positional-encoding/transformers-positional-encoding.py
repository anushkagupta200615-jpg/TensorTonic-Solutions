import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    positions = np.arange(seq_length).reshape(-1, 1)
    dims = np.arange(d_model).reshape(1, -1)
    angle_rates = positions / np.power(10000, (dims // 2) * 2 / d_model)
    pe = np.zeros_like(angle_rates, dtype=np.float64)
    pe[:, 0::2] = np.sin(angle_rates[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rates[:, 1::2])
    return pe
