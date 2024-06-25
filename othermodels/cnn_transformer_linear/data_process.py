from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

# create a custom dataset class
# wr is for image data and sp is for audio data
class MyDataset(Dataset):
    def __init__(self, x_wr, x_sp, y=None):
        self.x_wr = x_wr
        self.x_sp = x_sp
        self.y = y

    def __getitem__(self, index):
        # convert the numpy array to tensor
        features_wr = torch.tensor(self.x_wr[index], dtype = torch.float32).reshape(1, 28, 28)
        features_sp = torch.tensor(self.x_sp[index], dtype=torch.float32).unsqueeze(0)
        if self.y is not None:
            return features_wr, features_sp, torch.tensor(self.y[index], dtype=torch.long)
        else:
            return features_wr, features_sp

    def __len__(self):
        return len(self.x_wr)

# combine the image and audio data
# implement a data loader with batch size and validation size as arguments.
def data_loder(X_wr, X_sp, y, batch_size=64, val_size=0.2, seed=42):
    np.random.seed(seed)
    # split the data into training and validation
    data_size = len(X_wr)
    # shuffle the data
    indices = np.random.permutation(data_size)
    split = int(np.floor(val_size * data_size))
    # split the data
    X_train_wr, X_val_wr = X_wr[indices[split:]], X_wr[indices[:split]]
    X_train_sp, X_val_sp = X_sp[indices[split:]], X_sp[indices[:split]]
    y_train, y_val = y[indices[split:]], y[indices[:split]]

    # create the data loader
    train_loader = DataLoader(MyDataset(X_train_wr, X_train_sp, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MyDataset(X_val_wr, X_val_sp, y_val), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
