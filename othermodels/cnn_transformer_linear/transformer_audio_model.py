import torch.nn as nn
import torch
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _scaled_dot_product_attention(Q, K, V, mask=None):
    matmul_qk = torch.matmul(Q, K.transpose(-2, -1))

    # scale matmul_qk
    d_k = K.size(-1)
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, V)

    # output = output.transpose(1, 2).contiguous().view(64, -1, 3*d_k)
    return output, attention_weights

def generate_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.unsqueeze(0).unsqueeze(0) # Add batch dimension and another singleton dimension for compatibility

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model

        assert d_model % n_heads == 0

        self.depth = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.wo = nn.Linear(d_model, d_model)

    # split the d_model into n_heads
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.n_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = _scaled_dot_product_attention(q, k, v, mask)

        
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        scaled_attention = scaled_attention.view(batch_size, -1, self.d_model)

        output = self.wo(scaled_attention)

        return output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.mutlti_head_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None):
        attn_output, _ = self.mutlti_head_attention(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output) # residual connection
        x = self.norm1(x)

        # multi-head attention with encoder output
        ff_output = self.feed_forward(x)
        # add and norm
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout, n_classes=10):
        super(TransformerDecoder, self).__init__()

        self.n_layers = n_layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)])
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, tgt_mask=None):
        for layers in self.decoder_layers:
            x = layers(x, enc_output, tgt_mask)
        
        x = self.norm(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        
        if d_model % 2 == 1:  # check if d_model is odd
            # Make sure to include the last odd index if d_model is odd
            cos_div_term = torch.exp(torch.arange(0, d_model-1, 2).float() * (-np.log(10000.0) / d_model))
            self.encoding[:, 1::2] = torch.cos(position * cos_div_term)
        else:
            self.encoding[:, 1::2] = torch.cos(position * div_term)

        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', self.encoding)

    def forward(self, x):
        self.pe = self.pe.to(x.device)
        x = x + self.pe[:x.size(1), :]
        return x