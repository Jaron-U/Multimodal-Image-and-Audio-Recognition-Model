import torch
import torch.nn as nn
from multi_model_fusion import CNNTransformerAttention
import numpy as np
import pandas as pd
from data_process import data_loder
import matplotlib.pyplot as plt

# some parameters
batch_size = 128
lr = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 20

# transformer parameters
transformer_paras = {
    'd_model': 507,
    'n_heads': 13,
    'num_layers': 5,
    'dim_feedforward': 2048,
    'dropout': 0.1
}

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
    ax1.set_title('Loss Plot of CNN & Transformer Model')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(frameon=False)

    ax2.plot(train_acc, label='Training Accuracy')
    ax2.plot(val_acc, label='Validation Accuracy')
    ax2.set_title('Accuracy Plot of CNN & Transformer Model')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend(frameon=False)

    plt.tight_layout() 
    plt.savefig('loss_acc.png')


if __name__ == '__main__':
    X_train_wr = np.load('x_train_wr.npy')
    X_train_sp = np.load('x_train_sp.npy')
    y_train = pd.read_csv('y_train.csv')['label'].values
    
    # calculate the mean and standard deviation of the data
    mean_wr = X_train_wr.mean(axis=0)
    std_wr = X_train_wr.std(axis=0) + 1e-10
    x_train_wr = (X_train_wr - mean_wr) / std_wr

    mean_sp = X_train_sp.mean(axis=0)
    std_sp = X_train_sp.std(axis=0) + 1e-10
    x_train_sp = (X_train_sp - mean_sp) / std_sp

    train_loader, val_loader = data_loder(x_train_wr, x_train_sp, y_train, batch_size=batch_size)
    
    model = CNNTransformerAttention(cnn_kernel_size = 5, cnn_output_size=128, 
                                    num_classes=10, attention_paras=attention_paras, transformer_paras=transformer_paras)
    model, train_losses, val_losses, train_acc, val_acc= train_model(model, train_loader, val_loader)
    torch.save(model.state_dict(), 'cnn_transformer_attention.pth')
    plot_result(train_losses, val_losses, train_acc, val_acc)



'''
Epoch 1/20, Train Loss: 0.0507, Val Loss: 0.0699, Train Acc: 98.4292%, Val Acc: 97.9500%
Epoch 2/20, Train Loss: 0.0311, Val Loss: 0.0559, Train Acc: 99.0583%, Val Acc: 98.3833%
Epoch 3/20, Train Loss: 0.0170, Val Loss: 0.0499, Train Acc: 99.4479%, Val Acc: 98.5667%
Epoch 4/20, Train Loss: 0.0230, Val Loss: 0.0604, Train Acc: 99.2542%, Val Acc: 98.3250%
Epoch 5/20, Train Loss: 0.0110, Val Loss: 0.0438, Train Acc: 99.6250%, Val Acc: 98.8083%
Epoch 6/20, Train Loss: 0.0099, Val Loss: 0.0510, Train Acc: 99.6958%, Val Acc: 98.7000%
Epoch 7/20, Train Loss: 0.0073, Val Loss: 0.0503, Train Acc: 99.7521%, Val Acc: 98.7917%
Epoch 8/20, Train Loss: 0.0047, Val Loss: 0.0477, Train Acc: 99.8271%, Val Acc: 98.8000%
Epoch 9/20, Train Loss: 0.0085, Val Loss: 0.0508, Train Acc: 99.7229%, Val Acc: 98.8917%
Epoch 10/20, Train Loss: 0.0053, Val Loss: 0.0512, Train Acc: 99.8417%, Val Acc: 98.7917%
Epoch 11/20, Train Loss: 0.0044, Val Loss: 0.0515, Train Acc: 99.8667%, Val Acc: 98.9000%
Epoch 12/20, Train Loss: 0.0027, Val Loss: 0.0444, Train Acc: 99.9250%, Val Acc: 98.9167%
Epoch 13/20, Train Loss: 0.0082, Val Loss: 0.0700, Train Acc: 99.7208%, Val Acc: 98.4083%
Epoch 14/20, Train Loss: 0.0025, Val Loss: 0.0484, Train Acc: 99.9292%, Val Acc: 98.9333%
Epoch 15/20, Train Loss: 0.0042, Val Loss: 0.0599, Train Acc: 99.8521%, Val Acc: 98.8333%
Epoch 16/20, Train Loss: 0.0036, Val Loss: 0.0573, Train Acc: 99.8979%, Val Acc: 98.8333%
Epoch 17/20, Train Loss: 0.0022, Val Loss: 0.0522, Train Acc: 99.9146%, Val Acc: 98.9583%
Epoch 18/20, Train Loss: 0.0031, Val Loss: 0.0599, Train Acc: 99.8958%, Val Acc: 98.8000%
Epoch 19/20, Train Loss: 0.0018, Val Loss: 0.0525, Train Acc: 99.9500%, Val Acc: 98.9417%
Epoch 20/20, Train Loss: 0.0011, Val Loss: 0.0521, Train Acc: 99.9625%, Val Acc: 99.0000%
'''