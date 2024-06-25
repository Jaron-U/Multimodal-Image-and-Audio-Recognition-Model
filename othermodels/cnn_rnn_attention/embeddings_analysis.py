import pandas as pd
import torch
from data_process import data_loder
import numpy as np
from multi_model_fusion import CNNRNNAttentionFusion
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# some parameters
batch_size = 128
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

def extract_embeddings(model, X):
    model.eval()
    img_embeddings = []
    audio_embeddings = []
    labels = []

    with torch.no_grad():
        for batch, (x_wr, x_sp, y) in enumerate(X):
            x_wr, x_sp= x_wr.to(device), x_sp.to(device)
            y_pred, img_embedding, audio_embedding = model(x_wr, x_sp)
            
            img_embeddings.extend(img_embedding.cpu().numpy())
            audio_embeddings.extend(audio_embedding.cpu().numpy())
            labels.extend(y.cpu().numpy())

    return np.array(img_embeddings), np.array(audio_embeddings), np.array(labels)

def tsne_kmeans(img_embeddings, audio_embeddings, labels):
    # using tsne to reduce the dimensionality
    tsne = TSNE(n_components=2, random_state=42)
    img_tsne = tsne.fit_transform(img_embeddings)
    audio_tsne = tsne.fit_transform(audio_embeddings)

    # using kmeans to cluster the data
    kmeans = KMeans(n_clusters=10, random_state=42)
    img_clusters = kmeans.fit_predict(img_tsne)
    audio_clusters = kmeans.fit_predict(audio_tsne)

    # plot the data
    plot_clusters_comparison(img_tsne, img_clusters, labels, 'Image')
    plot_clusters_comparison(audio_tsne, audio_clusters, labels, 'Audio')

def plot_clusters_comparison(tsne_data, clusters, labels, title):
    flg, axs = plt.subplots(1, 2, figsize=(12, 6))

    # cluster plot
    scatter = axs[0].scatter(tsne_data[:, 0], tsne_data[:, 1], c=clusters, cmap='tab10')
    axs[0].set_title(f'{title}-Clusters')
    plt.colorbar(scatter, ax=axs[0])
    axs[0].grid(True)

    # label plot
    scatter = axs[1].scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels, cmap='tab10')
    axs[1].set_title(f'{title}-Labels')
    plt.colorbar(scatter, ax=axs[1])
    axs[1].grid(True)

    plt.savefig(f'{title}.png')

    
if __name__ == '__main__':
    # using train data to extract the embeddings
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

    train_loader, val_loader = data_loder(x_train_wr, x_train_sp, y_train, batch_size=batch_size, val_size=0.001)

    model = CNNRNNAttentionFusion(cnn_kernel_size = 3, rnn_num_layers = 2, cnn_output_size=128, 
                                  rnn_output_size=128, attention_paras=attention_paras)
    model.load_state_dict(torch.load('cnn_rnn_attention.pth'))
    img_embeddings, audio_embeddings, labels = extract_embeddings(model, train_loader)
    tsne_kmeans(img_embeddings, audio_embeddings, labels)


    

