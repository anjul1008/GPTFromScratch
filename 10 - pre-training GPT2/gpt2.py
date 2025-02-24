import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.esp = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdims=True)
        var = x.var(dim=-1, keepdims=True, unbiased=False)  #unbiased parameter to replicate GPT2, does devide by N not N-1
        x_norm = (x - mean) / torch.sqrt(var + self.esp)
        return self.scale * x_norm + self.scale 

# activation used in GPT-2 model
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
                                    torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                                    (x + 0.44715 * torch.pw(x, 3))
                                    )
                      )

# simple feedForwaed network
class FeedForward(nn.Module):
    def __init__(self, d_in, d_hidden=None):
        super().__init__()
        if d_hidden is None: d_hidden = 4 * d_in
        self.layers = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            GELU(),
            nn.Linear(d_hidden, d_in)
        )
    
    def forward(self, x):
        return self.layers(x)

class MHSA(nn.Module):
    def __init__(self, d_in, d_out, context_len, n_heads=2, causal=True, dropout=0.1, qkv_bias=False):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.causal = causal
        self.num_heads = n_heads
        self.head_dim = d_out // n_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        if self.causal:
            self.register_buffer('mask', torch.triu(torch.ones(context_len, context_len), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        
        # Original mask truncated to the number of tokens and converted to boolean
        if self.causal:
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            # Use the mask to fill attention scores
            attn_scores.masked_fill_(mask_bool, -torch.inf)
            # print('mask_bool.shape', mask_bool.shape)
            # print('attn_scores.shape', attn_scores.shape)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

class TransformerLayer(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.attn = MHSA(
            d_in = conf.get('emb_dim'),
            d_out = conf.get('emb_dim'),
            context_len = conf.get('context_len'),
            n_heads = conf.get('n_heads'),
            dropout = conf.get('drop_rate'),
            causal = True,
            qkv_bias = conf.get('qkv_bias')
        )
        
        self.ff = FeedForward(conf.get('emb_dim'))
        self.norm1 = LayerNorm(conf.get('emb_dim'))
        self.norm2 = LayerNorm(conf.get('emb_dim'))
        self.dropout = nn.Dropout(conf.get('drop_rate'))
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + shortcut
        # print(x)
        return x


class GPT2(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.tok_emb = nn.Embedding(conf.get('vocab_size'), conf.get('emb_dim'))
        self.pos_emb = nn.Embedding(conf.get('context_len'), conf.get('emb_dim'))
        self.drop_emb = nn.Dropout(conf.get('drop_rate'))
        
        self.trf_block = nn.Sequential(
            * [ TransformerLayer(conf) for _ in range(conf.get('n_layers')) ]
        )       # TODO: why * is needed here?
        
        self.final_norm = LayerNorm(conf.get('emb_dim'))
        self.out_head = nn.Linear(conf.get('emb_dim'), conf.get('vocab_size'), bias=False)
        
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=x.device))
        
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

