import torch
import torch.nn as nn
from cnn_img_model import CNNImgModel
from transformer_audio_model import TransformerDecoder, PositionalEncoding
import torch.nn.functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Assuming cnn_output_dim and rnn_output_dim are both set to 128
attention_paras = {
    'd_model': 128, # Output dimension of CNN and RNN
    'n_heads': 8,
    'num_layers': 1, # Number of attention layers
    'dropout': 0.1,
    'layer_norm': True,
    'residual': True
}

# transformer parameters
transformer_paras = {
    'd_model': 507,
    'n_heads': 13,
    'num_layers': 5,
    'dim_feedforward': 2048,
    'dropout': 0.1
}

def _scaled_dot_product_attention(Q, K, V, dropout=0.1, mask=None):
    matmul_qk = torch.matmul(Q, K.transpose(-2, -1))

    # scale matmul_qk
    d_k = K.size(-1)
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
    attention_weights = F.dropout(attention_weights, p=dropout)
    output = torch.matmul(attention_weights, V)

    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, layer_norm=True, residual=True, n_layers=1):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.layer_norm = layer_norm
        self.residual = residual
        self.n_layers = n_layers

        self.dropout_v = dropout
        self.dropout = nn.Dropout(dropout)

        assert d_model % n_heads == 0

        self.depth = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.wo = nn.Linear(d_model, d_model)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)

    # split the d_model into n_heads
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.n_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        residual = q

        for _ in range(self.n_layers):
            q = self.wq(q)
            k = self.wk(k)
            v = self.wv(v)

            q = self.split_heads(q, batch_size)
            k = self.split_heads(k, batch_size)
            v = self.split_heads(v, batch_size)

            scaled_attention, attention_weights = _scaled_dot_product_attention(q, k, v, self.dropout_v, mask)

            scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
            scaled_attention = scaled_attention.view(batch_size, -1, self.d_model)

            output = self.wo(scaled_attention)
            output = output.squeeze(1) # Remove the middle dimension (1)

            if self.residual:
                output += residual
                if self.layer_norm:
                    output = self.layer_norm(output)
            residual = output

        return output, attention_weights


class CNNTransformerAttention(nn.Module):
    def __init__(self, cnn_kernel_size = 3, rnn_num_layers = 2, cnn_output_size=128, rnn_output_size=128, 
                 num_classes=10, attention_paras=attention_paras, transformer_paras=transformer_paras):
        super(CNNTransformerAttention, self).__init__()
        self.d_model = transformer_paras['d_model'] 
        self.n_heads = transformer_paras['n_heads'] 
        self.num_decoder_layers = transformer_paras['num_layers']  
        self.dim_feedforward = transformer_paras['dim_feedforward']
        self.dropout = transformer_paras['dropout']
        self.cnn = CNNImgModel(kernel_size=cnn_kernel_size, num_classes=cnn_output_size).to(device)
        self.trans = TransformerDecoder(self.num_decoder_layers, self.d_model, self.n_heads, 
                                        self.dim_feedforward, self.dropout).to(device)
        self.pos = PositionalEncoding(self.d_model)
        
        self.multi_head_attention = MultiHeadAttention(attention_paras['d_model'], attention_paras['n_heads'], 
                                                       attention_paras['dropout'], attention_paras['layer_norm'], 
                                                       attention_paras['residual'], attention_paras['num_layers']).to(device)
        
        # create a linear layer for the combined output
        self.fc = nn.Linear(attention_paras['d_model'], num_classes)
        self.fc1 = nn.Linear(self.d_model, 128)
        self.to(device)
    
    def forward(self, x_wr, x_sp):
        # forward pass for the image data and audio data
        x_wr = self.cnn(x_wr)
        
        # forward pass for the image data and audio data
        x_sp = self.pos(x_sp)
        # x_mask = generate_mask(x_sp.size(0)).to(device)
        x_sp = self.trans(x_sp, x_sp)
        x_sp = self.fc1(x_sp)

        # getting embeddings for the audio and image data
        img_embedding = x_wr
        audio_embedding = x_sp[:, -1, :]

        # using the multi-head attention to fuse the outputs
        x, _ = self.multi_head_attention(x_wr, x_sp, x_sp) # Using the audio output as the query and key

        x = self.fc(x)
        return x, img_embedding, audio_embedding

