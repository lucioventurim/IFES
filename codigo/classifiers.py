# Classifiers and Scoring Definitions

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import numpy as np
import scipy.stats as stats
from sklearn.base import TransformerMixin
import pywt

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

def rms(x):
  '''
  root mean square
  '''
  x = np.array(x)
  return np.sqrt(np.mean(np.square(x)))

def sra(x):
  '''
  square root amplitude
  '''
  x = np.array(x)
  return np.mean(np.sqrt(np.absolute(x)))**2

def ppv(x):
  '''
  peak to peak value
  '''
  x = np.array(x)
  return np.max(x)-np.min(x)

def cf(x):
  '''
  crest factor
  '''
  x = np.array(x)
  return np.max(np.absolute(x))/rms(x)

def ifa(x):
  '''
  impact factor
  '''
  x = np.array(x)
  return np.max(np.absolute(x))/np.mean(np.absolute(x))

def mf(x):
  '''
  margin factor
  '''
  x = np.array(x)
  return np.max(np.absolute(x))/sra(x)

def sf(x):
  '''
  shape factor
  '''
  x = np.array(x)
  return rms(x)/np.mean(np.absolute(x))

def kf(x):
  '''
  kurtosis factor
  '''
  x = np.array(x)
  return stats.kurtosis(x)/(np.mean(x**2)**2)


class StatisticalTime(TransformerMixin):
  '''
  Extracts statistical features from the time domain.
  '''
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    return np.array([
                     [
                      rms(x), # root mean square
                      sra(x), # square root amplitude
                      stats.kurtosis(x), # kurtosis
                      stats.skew(x), # skewness
                      ppv(x), # peak to peak value
                      cf(x), # crest factor
                      ifa(x), # impact factor
                      mf(x), # margin factor
                      sf(x), # shape factor
                      kf(x), # kurtosis factor
                      ] for x in X[:,:]
                     ])


class StatisticalFrequency(TransformerMixin):
  '''
  Extracts statistical features from the frequency domain.
  '''
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    sig = []
    for x in X[:,:]:
      fx = np.absolute(np.fft.fft(x)) # transform x from time to frequency domain
      fc = np.mean(fx) # frequency center
      sig.append([
                  fc, # frequency center
                  rms(fx), # RMS from the frequency domain
                  rms(fx-fc), # Root Variance Frequency
                  ])
    return np.array(sig)

class Statistical(TransformerMixin):
  '''
  Extracts statistical features from both time and frequency domain.
  '''
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    st = StatisticalTime()
    stfeats = st.transform(X)
    sf = StatisticalFrequency()
    sffeats = sf.transform(X)
    return np.concatenate((stfeats,sffeats),axis=1)


from skimage.transform import resize


def create_cwt_images(X, n_scales, rescale_size, wavelet_name="morl"):
    n_samples = X.shape[0]
    n_signals = 1

    # range of scales from 1 to n_scales
    scales = np.arange(1, n_scales + 1)

    # pre allocate array
    X_cwt = np.ndarray(shape=(n_samples, rescale_size, rescale_size, n_signals), dtype='float32')


    for sample in range(n_samples):
        serie = X[sample]

        # continuous wavelet transform
        coeffs, freqs = pywt.cwt(serie, scales, wavelet_name)
        # resize the 2D cwt coeffs
        rescale_coeffs = resize(coeffs, (rescale_size, rescale_size), mode='constant')
        X_cwt[sample, :, :, 0] = rescale_coeffs

    return X_cwt



class Wavelet(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # amount of pixels in X and Y
        rescale_size = 227
        # determine the max scale size
        n_scales = 10

        X_cwt = create_cwt_images(X, n_scales, rescale_size)

        print(f"shapes (n_samples, x_img, y_img) of X_train_cwt: {X_cwt.shape}")

        return X_cwt

def Classifiers():

    # KNN
    from sklearn.neighbors import KNeighborsClassifier

    knn = Pipeline([
                    ('FeatureExtraction', Statistical()),
                    ('scaler', StandardScaler()),
                    ('knn', KNeighborsClassifier()),
                    ])

    parameters_knn = {'knn__n_neighbors': list(range(1,16,2))}

    knn = GridSearchCV(knn, parameters_knn)

    # Random Forest

    from sklearn.ensemble import RandomForestClassifier

    rf = Pipeline([
        ('FeatureExtraction', Statistical()),
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier()),
    ])

    parameters_rf = {
        "rf__max_features": [1, 5, 10],
        "rf__n_estimators": [10, 100, 200],
    }

    rf = GridSearchCV(rf, parameters_rf)

    # AlexNet

    # Function to create model, required for KerasClassifier
    def create_model():
        model = keras.models.Sequential([
            keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                                input_shape=(227, 227, 1)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='softmax')
        ])
        loss = 'binary_crossentropy'
        optimizer = 'adam'
        metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model

    alexnet = KerasClassifier(build_fn=create_model, verbose=0)

    alexnet_pipe = Pipeline([
        ('Wavelet', Wavelet()),
        ('alexnet', alexnet),
    ])


    # Chosen Classifiers

    clfs = [( 'AlexNet', alexnet_pipe), ('K-Nearest Neighbors', knn), ('Random Forest', rf)]

    return clfs

def Scoring():

    scoring = ['accuracy', 'f1_macro']

    return scoring
