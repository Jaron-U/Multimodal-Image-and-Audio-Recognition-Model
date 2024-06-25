import torch
import torch.nn as nn
from multi_model_fusion import CNNRNNAttentionFusion
import numpy as np
import pandas as pd
from data_process import data_loder
import matplotlib.pyplot as plt

# some parameters
batch_size = 128
lr = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 15

# Assuming cnn_output_dim and rnn_output_dim are both set to 128
attention_paras = {
    'd_model': 128, # Output dimension of CNN and RNN
    'n_heads': 8,
    'num_layers': 1, # Number of attention layers
    'dropout': 0.1,
    'layer_norm': True,
    'residual': True
}

def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    train_losses = []
    val_losses = []
    train_acc, val_acc = [], []
    for epoch in range(epochs):
        # train
        model.train()
        for batch, (x_wr, x_sp, y) in enumerate(train_loader):
            x_wr, x_sp, y = x_wr.to(device), x_sp.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred, _, _ = model(x_wr, x_sp)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        # evaluate
        # evluating the model on the training data
        model.eval()
        train_correct = 0
        train_total = 0
        epoch_train_losses = []
        with torch.no_grad():
            for batch, (x_wr, x_sp, y) in enumerate(train_loader):
                x_wr, x_sp, y = x_wr.to(device), x_sp.to(device), y.to(device)
                y_pred, _, _ = model(x_wr, x_sp)
                _, predicted = torch.max(y_pred, 1)
                train_total += y.size(0)
                train_correct += (predicted == y).sum().item()
                loss = criterion(y_pred, y)
                epoch_train_losses.append(loss.item())
            train_acc.append(100*train_correct / train_total)
            train_losses.append(np.mean(epoch_train_losses))

        model.eval()
        val_correct = 0
        val_total = 0
        epoch_val_losses = []
        with torch.no_grad():
            for batch, (x_wr, x_sp, y) in enumerate(val_loader):
                x_wr, x_sp, y = x_wr.to(device), x_sp.to(device), y.to(device)
                y_pred, _, _ = model(x_wr, x_sp)
                loss = criterion(y_pred, y)
                epoch_val_losses.append(loss.item())
                _, predicted = torch.max(y_pred, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        val_acc.append(100*val_correct / val_total)
        val_losses.append(np.mean(epoch_val_losses))

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, " 
              f"Train Acc: {train_acc[-1]:.4f}%, Val Acc: {val_acc[-1]:.4f}%")
    return model, train_losses, val_losses, train_acc, val_acc

def plot_result(train_losses, val_losses, train_acc, val_acc):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Loss Plot of CNN & RNN Model')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(frameon=False)

    ax2.plot(train_acc, label='Training Accuracy')
    ax2.plot(val_acc, label='Validation Accuracy')
    ax2.set_title('Accuracy Plot of CNN & RNN Model')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend(frameon=False)

    plt.tight_layout() 
    plt.savefig('loss_acc.png')


if __name__ == '__main__':
    X_train_wr = np.load('../x_train_wr.npy')
    X_train_sp = np.load('../x_train_sp.npy')
    y_train = pd.read_csv('../y_train.csv')['label'].values

    # calculate the mean and standard deviation of the data
    mean_wr = X_train_wr.mean(axis=0)
    std_wr = X_train_wr.std(axis=0) + 1e-10
    x_train_wr = (X_train_wr - mean_wr) / std_wr

    mean_sp = X_train_sp.mean(axis=0)
    std_sp = X_train_sp.std(axis=0) + 1e-10
    x_train_sp = (X_train_sp - mean_sp) / std_sp

    train_loader, val_loader = data_loder(x_train_wr, x_train_sp, y_train, batch_size=batch_size)
    
    model = CNNRNNAttentionFusion(cnn_kernel_size = 3, rnn_num_layers = 2, cnn_output_size=128, 
                                  rnn_output_size=128, attention_paras=attention_paras)
    model, train_losses, val_losses, train_acc, val_acc= train_model(model, train_loader, val_loader)
    torch.save(model.state_dict(), 'cnn_rnn_attention.pth')
    plot_result(train_losses, val_losses, train_acc, val_acc)



'''
Epoch 1/15, Train Loss: 0.0463, Val Loss: 0.0674, Train Acc: 98.5792%, Val Acc: 97.9500%
Epoch 2/15, Train Loss: 0.0262, Val Loss: 0.0514, Train Acc: 99.2000%, Val Acc: 98.5167%
Epoch 3/15, Train Loss: 0.0147, Val Loss: 0.0435, Train Acc: 99.5479%, Val Acc: 98.8167%
Epoch 4/15, Train Loss: 0.0217, Val Loss: 0.0567, Train Acc: 99.3083%, Val Acc: 98.6083%
Epoch 5/15, Train Loss: 0.0086, Val Loss: 0.0427, Train Acc: 99.7375%, Val Acc: 98.8500%
Epoch 6/15, Train Loss: 0.0069, Val Loss: 0.0449, Train Acc: 99.7917%, Val Acc: 98.8250%
Epoch 7/15, Train Loss: 0.0085, Val Loss: 0.0532, Train Acc: 99.6937%, Val Acc: 98.7833%
Epoch 8/15, Train Loss: 0.0064, Val Loss: 0.0492, Train Acc: 99.8021%, Val Acc: 98.8417%
Epoch 9/15, Train Loss: 0.0046, Val Loss: 0.0442, Train Acc: 99.8542%, Val Acc: 98.9583%
Epoch 10/15, Train Loss: 0.0031, Val Loss: 0.0377, Train Acc: 99.9104%, Val Acc: 99.0833%
Epoch 11/15, Train Loss: 0.0019, Val Loss: 0.0471, Train Acc: 99.9604%, Val Acc: 99.0417%
Epoch 12/15, Train Loss: 0.0045, Val Loss: 0.0540, Train Acc: 99.8542%, Val Acc: 98.8250%
Epoch 13/15, Train Loss: 0.0045, Val Loss: 0.0492, Train Acc: 99.8563%, Val Acc: 98.9250%
Epoch 14/15, Train Loss: 0.0036, Val Loss: 0.0494, Train Acc: 99.8521%, Val Acc: 98.9417%
Epoch 15/15, Train Loss: 0.0016, Val Loss: 0.0516, Train Acc: 99.9562%, Val Acc: 98.8750%
'''
