# Random Forest

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from features_extractors.statistical import Statistical


def instantiate_auto_random_forest():

    rf = Pipeline([
        ('FeatureExtraction', Statistical()),
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier()),
    ])

    parameters_rf = {
        "rf__max_features": [1, 2, 5, 7, 10],
        "rf__n_estimators": [50, 100, 200],
    }

    rf = GridSearchCV(rf, parameters_rf)

    return rf
