from typing import List
import torch


def batch(sentences_input: List[any], batch_size: int):
    return [sentences_input[x:x + batch_size] for x in range(0, len(sentences_input), batch_size)]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
