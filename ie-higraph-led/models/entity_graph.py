import torch
import torch.nn as nn
import networkx as nx

class EntityGraphLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=8)

    def forward(self, entity_embeddings):
        # entity_embeddings: [num_entities, batch, hidden]
        attn_output, _ = self.attn(entity_embeddings,
                                   entity_embeddings,
                                   entity_embeddings)
        return attn_output
