# SVM

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from features_extractors.statistical import Statistical
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin


class SVM(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.svm = SVC()

    def fit(self, X, y=None):
        self.svm.fit(X, y)

    def predict_proba(self, X, y=None):
        return self.svm.decision_function(X)

    def predict(self, X, y=None):
        return self.svm.predict(X)


def instantiate_auto_svm():

    svm = Pipeline([
                    ('FeatureExtraction', Statistical()),
                    ('scaler', StandardScaler()),
                    #('svm', SVC(probability="True")),
                    ('svm', SVM()),
                    ])

    return svm
