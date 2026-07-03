import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray,
               gamma: np.ndarray,
               beta: np.ndarray,
               eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)
    return gamma * normalized + beta

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, V)

def multi_head_attention(Q, K, V,
                         W_q, W_k, W_v, W_o,
                         num_heads: int) -> np.ndarray:
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    Q_proj = np.matmul(Q, W_q)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)

    def split_heads(x):
        return x.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    Q_heads = split_heads(Q_proj)
    K_heads = split_heads(K_proj)
    V_heads = split_heads(V_proj)

    head_outputs = []
    for h in range(num_heads):
        Q_h = Q_heads[:, h, :, :]
        K_h = K_heads[:, h, :, :]
        V_h = V_heads[:, h, :, :]
        head_outputs.append(scaled_dot_product_attention(Q_h, K_h, V_h))

    concat = np.stack(head_outputs, axis=1)
    concat = concat.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    return np.matmul(concat, W_o)

def feed_forward(x, W1, b1, W2, b2):
    hidden = np.dot(x, W1) + b1
    relu_out = np.maximum(0, hidden)
    return np.dot(relu_out, W2) + b2

def encoder_block(x,
                  W_q, W_k, W_v, W_o,
                  W1, b1, W2, b2,
                  gamma1, beta1,
                  gamma2, beta2,
                  num_heads: int) -> np.ndarray:
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x1 = layer_norm(x + attn_out, gamma1, beta1)
    ff_out = feed_forward(x1, W1, b1, W2, b2)
    output = layer_norm(x1 + ff_out, gamma2, beta2)
    return output
