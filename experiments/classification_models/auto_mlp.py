# MLP

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from features_extractors.heterogeneous import Heterogeneous
from sklearn.neural_network import MLPClassifier


def instantiate_auto_mlp():

    mlp = Pipeline([
                    ('FeatureExtraction', Heterogeneous()),
                    ('scaler', StandardScaler()),
                    ('mlp', MLPClassifier(max_iter=500)),
                    ])

    return mlp