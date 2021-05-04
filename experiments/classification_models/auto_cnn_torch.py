
from torch import nn
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from skorch import NeuralNetClassifier


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

# ----------------------------inputsize >=28-------------------------------------------------------------------------
class CNN_torch_model(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=3):
        super(CNN_torch_model, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=15),  # 16, 26 ,26
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3),  # 32, 24, 24
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))  # 32, 12,12     (24-2) /2 +1

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),  # 64,10,10
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),  # 128,8,8
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4))  # 128, 4,4

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True))
        self.fc = nn.Linear(64, out_channel)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        x = self.fc(x)

        return x

class CNN(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 optimizer='adam',
                 epochs=10
                 ):
        self.optimizer = optimizer
        self.epochs = epochs
        self.n_features = 1

    def fit(self, X, y=None):
        optimizer = self.optimizer
        epochs = self.epochs

        X = X.astype(np.float32)

        # Define innput shapes
        self.n_samples = X.shape[0]
        self.n_steps = X.shape[1]
        X = X.reshape((X.shape[0], self.n_features, X.shape[1]))


        self.labels, ids = np.unique(y, return_inverse=True)
        num_classes = len(self.labels)
        y_cat = to_categorical(ids, num_classes)

        self.model = NeuralNetClassifier(
            CNN_torch_model(in_channel=self.n_features, out_channel=num_classes),
            max_epochs=self.epochs,
            lr=0.1,
            train_split=None
            )

        self.model.fit(X, y_cat)


    def predict_proba(self, X, y=None):
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        return self.model.predict(X)

    def predict(self, X, y=None):
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        predictions = self.model.predict(X)
        return self.labels[np.argmax(predictions, axis=1)]

def instantiate_auto_cnn():


    cnn = CNN()


    return cnn