import torch
import torch.nn as nn
from cnn_img_model import CNNImgModel
from transformer_audio_model import TransformerDecoder, PositionalEncoding

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# transformer parameters
transformer_paras = {
    'd_model': 507,
    'n_heads': 13,
    'num_layers': 5,
    'dim_feedforward': 2048,
    'dropout': 0.1
}

class CNNTransformerLinear(nn.Module):
    def __init__(self, transformer_paras=transformer_paras, cnn_kernel_size = 3, cnn_output_size=10, num_classes=10):
        super(CNNTransformerLinear, self).__init__()
        self.d_model = transformer_paras['d_model'] 
        self.n_heads = transformer_paras['n_heads'] 
        self.num_decoder_layers = transformer_paras['num_layers']  
        self.dim_feedforward = transformer_paras['dim_feedforward']
        self.dropout = transformer_paras['dropout']

        self.cnn = CNNImgModel(kernel_size=cnn_kernel_size, num_classes=cnn_output_size).to(device)
        self.trans = TransformerDecoder(self.num_decoder_layers, self.d_model, self.n_heads, 
                                        self.dim_feedforward, self.dropout).to(device)
        
        self.pos = PositionalEncoding(self.d_model)
        
        self.fc1 = nn.Linear(cnn_output_size+self.d_model, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)
        self.to(device)

    def forward(self, x_wr, x_sp):
        # forward pass for the image data and audio data
        x_sp = self.pos(x_sp)
        # x_mask = generate_mask(x_sp.size(0)).to(device)
        x_sp = self.trans(x_sp, x_sp)

        x_wr = self.cnn(x_wr)

        # getting embeddings for the audio and image data
        img_embedding = x_wr
        audio_embedding = x_sp[:, -1, :]

        combined_output = torch.cat((x_wr, x_sp[:, -1, :]), dim=1)
        x = self.fc1(combined_output)
        x = self.relu(x)
        x = self.fc2(x)
        return x, img_embedding, audio_embedding

