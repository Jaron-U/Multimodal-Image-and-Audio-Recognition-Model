import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_process import MyDataset
import numpy as np
from multi_model_fusion import CNNRNNAttentionFusion

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128

# Assuming cnn_output_dim and rnn_output_dim are both set to 128
attention_paras = {
    'd_model': 128, # Output dimension of CNN and RNN
    'n_heads': 8,
    'num_layers': 1, # Number of attention layers
    'dropout': 0.1,
    'layer_norm': True,
    'residual': True
}

def predict(model, X):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch, (x_wr, x_sp) in enumerate(X):
            x_wr, x_sp= x_wr.to(device), x_sp.to(device)
            y_pred, _, _ = model(x_wr, x_sp)
            _, y_pred_labels = torch.max(y_pred, dim=1)
            predictions.extend(y_pred_labels.cpu().numpy())
    return predictions
    

if __name__ == '__main__':
    X_test_wr = np.load('../x_test_wr.npy')
    X_test_sp = np.load('../x_test_sp.npy')

    # calculate the mean and standard deviation of the data
    mean_wr = X_test_wr.mean(axis=0)
    std_wr = X_test_wr.std(axis=0) + 1e-10
    x_test_wr = (X_test_wr - mean_wr) / std_wr

    mean_sp = X_test_sp.mean(axis=0)
    std_sp = X_test_sp.std(axis=0) + 1e-10
    x_test_sp = (X_test_sp - mean_sp) / std_sp

    test_loader = DataLoader(MyDataset(x_test_wr, x_test_sp), batch_size=batch_size, shuffle=False)

    model = CNNRNNAttentionFusion(cnn_kernel_size = 3, rnn_num_layers = 2, cnn_output_size=128, 
                                  rnn_output_size=128, attention_paras=attention_paras)
    model.load_state_dict(torch.load('cnn_rnn_attention.pth'))
    predictions = predict(model, test_loader)
    # convert the output to a pandas dataframe
    df = pd.DataFrame({'label': predictions})
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'row_id'}, inplace=True)

    df.to_csv('prediction.csv', index=False)


    