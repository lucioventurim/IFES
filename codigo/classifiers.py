# Classifiers and Scoring Definitions

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import numpy as np
import scipy.stats as stats
from sklearn.base import TransformerMixin

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

def Classifiers():

    # KNN
    from sklearn.neighbors import KNeighborsClassifier

    knn = Pipeline([
                    ('FeatureExtraction', StatisticalTime()),
                    ('scaler', StandardScaler()),
                    ('knn', KNeighborsClassifier()),
                    ])

    parameters_knn = {'knn__n_neighbors': list(range(1,16,2))}

    knn = GridSearchCV(knn, parameters_knn)

    # SVM
    from sklearn.svm import SVC

    svm = Pipeline([
                    ('FeatureExtraction', StatisticalTime()),
                    ('scaler', StandardScaler()),
                    ('svc', SVC()),
                    ])

    parameters_svm = {
        'svc__C': [10**x for x in range(-1,2)],
        'svc__gamma': [10**x for x in range(-2,1)],
        }

    svm = GridSearchCV(svm, parameters_svm)

    # MLP
    from sklearn.neural_network import MLPClassifier

    mlp = Pipeline([
                    ('FeatureExtraction', StatisticalTime()),
                    ('scaler', StandardScaler()),
                    ('mlp', MLPClassifier()),
                    ])

    parameters_mlp = {
        'mlp__solver': ['lbfgs'],
        'mlp__alpha': [1e-5],
        'mlp__hidden_layer_sizes' : [(5, 2)],
        }

    mlp = GridSearchCV(mlp, parameters_mlp)

    # Random Forest

    from sklearn.ensemble import RandomForestClassifier

    rf = Pipeline([
        ('FeatureExtraction', StatisticalTime()),
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier()),
    ])

    parameters_rf = {
        "rf__max_features": [1, 5, 10],
        "rf__n_estimators": [10, 100, 200],
    }

    rf = GridSearchCV(rf, parameters_rf)



    # Chosen Classifiers

    clfs = [('K-Nearest Neighbors', knn)]#, ('Random Forest', rf), ('MLP', mlp), ('SVM', svm)]

    return clfs

def Scoring():

    scoring = ['accuracy', 'f1_macro']

    return scoring
