import torch
import torch.nn as nn
from multi_model_fusion import CNNTransformerLinear
import numpy as np
import pandas as pd
from data_process import data_loder
import matplotlib.pyplot as plt

# some parameters
batch_size = 125
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
    
    model = CNNTransformerLinear(transformer_paras=transformer_paras, cnn_kernel_size = 3, 
                                 cnn_output_size=10, num_classes=10)
    model, train_losses, val_losses, train_acc, val_acc= train_model(model, train_loader, val_loader)
    torch.save(model.state_dict(), 'cnn_transformer_linear.pth')
    plot_result(train_losses, val_losses, train_acc, val_acc)



'''
Epoch 1/20, Train Loss: 0.0861, Val Loss: 0.1053, Train Acc: 97.2417%, Val Acc: 96.6250%
Epoch 2/20, Train Loss: 0.0366, Val Loss: 0.0563, Train Acc: 98.8396%, Val Acc: 98.2083%
Epoch 3/20, Train Loss: 0.0375, Val Loss: 0.0668, Train Acc: 98.8500%, Val Acc: 98.0250%
Epoch 4/20, Train Loss: 0.0195, Val Loss: 0.0440, Train Acc: 99.3958%, Val Acc: 98.5417%
Epoch 5/20, Train Loss: 0.0195, Val Loss: 0.0502, Train Acc: 99.3792%, Val Acc: 98.5000%
Epoch 6/20, Train Loss: 0.0165, Val Loss: 0.0525, Train Acc: 99.4542%, Val Acc: 98.5333%
Epoch 7/20, Train Loss: 0.0244, Val Loss: 0.0700, Train Acc: 99.2229%, Val Acc: 98.2500%
Epoch 8/20, Train Loss: 0.0125, Val Loss: 0.0659, Train Acc: 99.5625%, Val Acc: 98.4583%
Epoch 9/20, Train Loss: 0.0072, Val Loss: 0.0506, Train Acc: 99.7708%, Val Acc: 98.7333%
Epoch 10/20, Train Loss: 0.0129, Val Loss: 0.0586, Train Acc: 99.6521%, Val Acc: 98.6417%
Epoch 11/20, Train Loss: 0.0069, Val Loss: 0.0554, Train Acc: 99.7812%, Val Acc: 98.6833%
Epoch 12/20, Train Loss: 0.0067, Val Loss: 0.0632, Train Acc: 99.7833%, Val Acc: 98.7000%
Epoch 13/20, Train Loss: 0.0067, Val Loss: 0.0618, Train Acc: 99.8042%, Val Acc: 98.6083%
Epoch 14/20, Train Loss: 0.0029, Val Loss: 0.0503, Train Acc: 99.9188%, Val Acc: 98.8833%
Epoch 15/20, Train Loss: 0.0020, Val Loss: 0.0589, Train Acc: 99.9313%, Val Acc: 98.8083%
Epoch 16/20, Train Loss: 0.0029, Val Loss: 0.0605, Train Acc: 99.9000%, Val Acc: 98.7917%
Epoch 17/20, Train Loss: 0.0025, Val Loss: 0.0619, Train Acc: 99.9208%, Val Acc: 98.8917%
Epoch 18/20, Train Loss: 0.0021, Val Loss: 0.0502, Train Acc: 99.9333%, Val Acc: 98.9333%
Epoch 19/20, Train Loss: 0.0031, Val Loss: 0.0686, Train Acc: 99.9062%, Val Acc: 98.8250%
Epoch 20/20, Train Loss: 0.0009, Val Loss: 0.0560, Train Acc: 99.9813%, Val Acc: 98.8583%
'''