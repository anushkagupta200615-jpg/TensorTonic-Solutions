import numpy as np
from typing import Tuple

def apply_mlm_mask(
    token_ids: np.ndarray,
    mask_positions: np.ndarray,
    replace_probs: np.ndarray,
    random_tokens: np.ndarray,
    mask_token_id: int = 103
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply BERT's 80-10-10 masking strategy.

    Returns:
        masked_ids: token IDs after masking
        labels: original IDs at masked positions, -100 elsewhere
    """
    masked_ids = token_ids.copy()
    labels = np.full_like(token_ids, fill_value=-100)

    batch_size, seq_len = token_ids.shape
    for i in range(batch_size):
        for j in range(seq_len):
            if mask_positions[i, j]:
                labels[i, j] = token_ids[i, j]  # store original token
                prob = replace_probs[i, j]
                if prob < 0.8:
                    masked_ids[i, j] = mask_token_id
                elif prob < 0.9:
                    masked_ids[i, j] = random_tokens[i, j]
                else:
                    masked_ids[i, j] = token_ids[i, j]  # unchanged
    return masked_ids, labels


class MLMHead:
    """Masked LM prediction head."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token logits: hidden_states @ W + b

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        return hidden_states @ self.W + self.b
