# ResNet

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

import tensorflow.keras as keras



class ResNet(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 kernel_size=32,
                 filters=32,
                 optimizer='adam',
                 epochs=10
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

        n_feature_maps = 64

        input_shape = X.shape[1:]
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=6, kernel_size=7, padding='valid', activation='sigmoid')(input_layer)
        conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)

        conv2 = keras.layers.Conv1D(filters=12, kernel_size=7, padding='valid', activation='sigmoid')(conv1)
        conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

        flatten_layer = keras.layers.Flatten()(conv2)

        output_layer = keras.layers.Dense(units=num_classes, activation='sigmoid')(flatten_layer)

        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['categorical_accuracy'])

    def predict_proba(self, X, y=None):
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        return self.model.predict(X)

    def predict(self, X, y=None):
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        predictions = self.model.predict(X)
        return self.labels[np.argmax(predictions, axis=1)]


def instantiate_auto_resnet():

    resnet = ResNet()

    return resnet
