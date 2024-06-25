import torch
import torch.nn as nn
from cnn_img_model import CNNImgModel
from rnn_audio_model import LSTMAudioModel
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CNNRNNLinear(nn.Module):
    def __init__(self, cnn_kernel_size = 3, rnn_num_layers = 2, cnn_output_size=10, rnn_output_size=10, num_classes=10):
        super(CNNRNNLinear, self).__init__()
        
        self.cnn = CNNImgModel(kernel_size=cnn_kernel_size, num_classes=cnn_output_size).to(device)
        self.rnn = LSTMAudioModel(num_layers=rnn_num_layers, num_classes=rnn_output_size).to(device)

        # create a linear layer for the combined output
        self.fc = nn.Linear(cnn_output_size + rnn_output_size, num_classes)
        self.to(device)
    
    def forward(self, x_wr, x_sp):
        # forward pass for the image data and audio data
        x_wr = self.cnn(x_wr)
        x_sp = self.rnn(x_sp)

        # getting embeddings for the audio and image data
        img_embedding = x_wr
        audio_embedding = x_sp

        # concatenate the output from the cnn and rnn
        x = torch.cat((x_wr, x_sp), dim=1)
        x = self.fc(x)
        return x, img_embedding, audio_embedding

