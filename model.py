import torch
import torch.nn as nn
from config import INIT_STD

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_heads, depth, max_seq_len=2048):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads, dim_feedforward=4*emb_dim,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)

        # Tie weights
        self.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Applies the default scaled initialization."""
        init_val = (2**0.5) * INIT_STD
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_val)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_val)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)

        x = self.token_embedding(x) + self.pos_embedding(positions)

        # Causal mask for autoregressive generation
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.norm(x)
        return self.lm_head(x)