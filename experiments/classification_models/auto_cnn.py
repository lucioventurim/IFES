# CNN

from sklearn.pipeline import Pipeline
from tensorflow import keras
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.preprocessing import LabelEncoder


class CNN(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.n_features = 1
        self.n_samples = 0
        self.n_steps = 8192
        self.encoder = LabelEncoder()

    # Function to create model, required for KerasClassifier
    def create_model(self):
        model_m = keras.Sequential()
        model_m.add(keras.layers.Conv1D(100, 10, activation='relu', input_shape=(self.n_steps, self.n_features)))
        model_m.add(keras.layers.Flatten())
        model_m.add(keras.layers.Dense(3, activation='softmax'))
        #print(model_m.summary())

        loss = 'categorical_crossentropy'
        optimizer = 'adam'
        metrics = ['accuracy']
        model_m.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model_m

    def fit(self, X, y=None):

        # Define innput shapes
        self.n_samples = X.shape[0]
        self.n_steps = X.shape[1]
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))

        # Encode categorical variables
        self.encoder.fit(y)
        encoded_y = self.encoder.transform(y)
        dummy_y = keras.utils.to_categorical(encoded_y)

        self.cnn = keras.wrappers.scikit_learn.KerasClassifier(build_fn=self.create_model, verbose=0)
        self.cnn.fit(X, dummy_y)

    def predict(self, X, y=None):

        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        y_pred = self.cnn.predict(X)

        y_pred = self.encoder.inverse_transform(y_pred)

        return y_pred

    def predict_proba(self, X, y=None):
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        y_pred_proba = self.cnn.predict_proba(X)

        return y_pred_proba


def instantiate_auto_cnn():

    cnn = CNN()

    return cnn
