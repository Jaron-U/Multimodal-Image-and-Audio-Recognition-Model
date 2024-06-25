# Multimodal Image and Audio Recognition Model
In this project, a multimodal digital recognition model was developed that combines a Convolutional Neural Network (CNN) and a Transformer-based decoder to effectively process digital audio and image data through a multi-head attention mechanism. Utilizing CNN to analyze image data of 28x28 pixels and Transformer to train audio vectors of length 507, this model demonstrates superior training speed and convergence compared to traditional CNN and RNN methods. The model achieved an accuracy of 99.5% in the Kaggle competition.

## Result
I have experimented with various model architectures and hyperparameters to achieve the best performance. The final model achieved an accuracy of 99.5% in the Kaggle competition.

### Model1: CNN+RNN with Linear Fusion
* Loss and Accuracy Plot
![Model1](/othermodels/cnn_rnn_linear/loss_acc.png){:width="50%"}
* Train Loss: 0.0027, Val Loss: 0.0593, Train Acc: 99.9125%, Val Acc: 98.8750%

### Model2: CNN+RNN with Attention Fusion
* Loss and Accuracy Plot
![Model2](/othermodels/cnn_rnn_attention/loss_acc.png){:width="50%"}
* Train Loss: Train Loss: 0.0016, Val Loss: 0.0516, Train Acc: 99.9562%, Val Acc: 98.8750%

### Model3: CNN+Transformer with Linear Fusion
![Model3](/othermodels/cnn_transformer_linear/loss_acc.png){:width="50%"}
* Train Loss: Train Loss: 0.0009, Val Loss: 0.0560, Train Acc: 99.9813%, Val Acc: 98.8583%

### Model4: CNN+Transformer with Attention Fusion
![Model4](/loss_acc.png){:width="50%"}
* Train Loss: Train Loss: 0.0011, Val Loss: 0.0521, Train Acc: 99.9625%, Val Acc: 99.0000%

These models are close in accuracy and converge very fast

## TSNE-Kmeans
I used t-SNE to reduce the image and audio embedding data to 2-dimensional space. Color code each point by the corresponding label. Apply kmeans clustering with k = 10.
![TSNE-Kmeans](/Image.png){:width="50%"}
![TSNE-Kmeans](/Audio.png){:width="50%"}

For image data, the clustering results are quite clear, with almost each cluster corresponding to a separate region. The model is capable of features that distinguish different digits in image data. However, for the audio data, there is no clear one-to-one correspondence between clusters and real labels. This is due to the fact that the transformer model I’m using doesn’t capture the features of the audio data very well. So it leads to poor model performance. Another reason is that I hardly did any pre-processing on the audio data.

## Requirements
- Python 3.10
- torch 2.2.2
- matplotlib
- numpy
- pandas
- scikit-learn

## How to run
1. add the `x_train_wr.npy`, `x_train_sp.npy`, `x_test_wr.npy`, `x_test_sp.npy`, and `y_train.csv` files to the current folder.
2. run the `train.py` file. And the model will be saved in the `cnn_transformer_attention.pth`.
3. run the `predict.py` file. And the result will be saved in the `predict.py`.