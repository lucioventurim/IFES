# CNN

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class CNN(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 kernel_size=32,
                 filters=32,
                 optimizer='adam',
                 epochs=100
                 ):
        self.kernel_size = kernel_size
        self.filters = filters
        self.optimizer = optimizer
        self.epochs = epochs
        self.n_features = 1

    def fit(self, X, y=None):
        kernel_size = self.kernel_size
        filters = self.filters
        optimizer = self.optimizer
        epochs = self.epochs

        # Define innput shapes
        self.n_samples = X.shape[0]
        self.n_steps = X.shape[1]
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))

        self.labels, ids = np.unique(y, return_inverse=True)
        y_cat = to_categorical(ids)
        num_classes = y_cat.shape[1]

        self.model = Sequential()
        self.model.add(layers.InputLayer(input_shape=(self.n_steps, self.n_features)))
        self.model.add(layers.Conv1D(filters, kernel_size))
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(num_classes))
        self.model.add(layers.Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=["categorical_accuracy"])
        self.model.fit(X, y_cat, epochs=epochs, verbose=False)

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
