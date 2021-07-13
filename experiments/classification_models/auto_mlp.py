# MLP

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from features_extractors.statistical import Statistical
from sklearn.neural_network import MLPClassifier


def instantiate_auto_mlp():

    mlp = Pipeline([
                    ('FeatureExtraction', Statistical()),
                    ('scaler', StandardScaler()),
                    ('mlp', MLPClassifier(max_iter=500)),
                    ])

    return mlp