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

        # BLOCK 1

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(num_classes, activation='softmax')(gap_layer)

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
