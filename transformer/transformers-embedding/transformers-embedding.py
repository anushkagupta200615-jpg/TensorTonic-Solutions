import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    torch.manual_seed(42)
    embedding=nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    return embedding



def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    token_tensor = torch.tensor(tokens, dtype=torch.long)
    raw_embeddings = embedding(token_tensor)
    scaled_embeddings = raw_embeddings * math.sqrt(d_model)
    return scaled_embeddings
