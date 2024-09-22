import torch
import torch.nn as nn
from transformer import TransformerBlock, MultiHeadAttention, FeedForward, GPT_CONFIG_124M
from layer_norm import LayerNorm

class GPTModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["embedding_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["num_layers"])]
        )
        self.final_norm = LayerNorm(config["embedding_dim"])
        self.out_head = nn.Linear(
            config["embedding_dim"], config["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
