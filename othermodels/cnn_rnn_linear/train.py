import torch
import torch.nn as nn
from multi_model_fusion import CNNRNNLinear
import numpy as np
import pandas as pd
from data_process import data_loder
import matplotlib.pyplot as plt

# some parameters
batch_size = 64
lr = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 15

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
    
    model = CNNRNNLinear(cnn_kernel_size = 3, rnn_num_layers = 3, cnn_output_size=10, rnn_output_size=10)
    model, train_losses, val_losses, train_acc, val_acc= train_model(model, train_loader, val_loader)
    torch.save(model.state_dict(), 'cnn_rnn_linear.pth')
    plot_result(train_losses, val_losses, train_acc, val_acc)


'''
Epoch 1/15, Train Loss: 0.0650, Val Loss: 0.0854, Train Acc: 98.0375%, Val Acc: 97.3667%
Epoch 2/15, Train Loss: 0.0336, Val Loss: 0.0543, Train Acc: 98.9792%, Val Acc: 98.4083%
Epoch 3/15, Train Loss: 0.0295, Val Loss: 0.0528, Train Acc: 99.0833%, Val Acc: 98.3833%
Epoch 4/15, Train Loss: 0.0184, Val Loss: 0.0486, Train Acc: 99.3896%, Val Acc: 98.5250%
Epoch 5/15, Train Loss: 0.0136, Val Loss: 0.0514, Train Acc: 99.5438%, Val Acc: 98.6083%
Epoch 6/15, Train Loss: 0.0084, Val Loss: 0.0420, Train Acc: 99.7542%, Val Acc: 98.7917%
Epoch 7/15, Train Loss: 0.0099, Val Loss: 0.0499, Train Acc: 99.7062%, Val Acc: 98.7667%
Epoch 8/15, Train Loss: 0.0053, Val Loss: 0.0437, Train Acc: 99.8521%, Val Acc: 98.9333%
Epoch 9/15, Train Loss: 0.0066, Val Loss: 0.0485, Train Acc: 99.8250%, Val Acc: 98.8250%
Epoch 10/15, Train Loss: 0.0075, Val Loss: 0.0487, Train Acc: 99.8063%, Val Acc: 98.7917%
Epoch 11/15, Train Loss: 0.0074, Val Loss: 0.0480, Train Acc: 99.7646%, Val Acc: 98.8417%
Epoch 12/15, Train Loss: 0.0113, Val Loss: 0.0594, Train Acc: 99.6542%, Val Acc: 98.6333%
Epoch 13/15, Train Loss: 0.0046, Val Loss: 0.0534, Train Acc: 99.8438%, Val Acc: 98.9000%
Epoch 14/15, Train Loss: 0.0024, Val Loss: 0.0490, Train Acc: 99.9208%, Val Acc: 99.0583%
Epoch 15/15, Train Loss: 0.0027, Val Loss: 0.0593, Train Acc: 99.9125%, Val Acc: 98.8750%
'''