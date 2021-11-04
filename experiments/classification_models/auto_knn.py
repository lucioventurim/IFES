# KNN

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from features_extractors.heterogeneous import Heterogeneous
from sklearn.neighbors import KNeighborsClassifier


def instantiate_auto_knn():

    knn = Pipeline([
                    ('FeatureExtraction', Heterogeneous()),
                    ('scaler', StandardScaler()),
                    ('knn', KNeighborsClassifier()),
                    ])

    #parameters_knn = {'knn__n_neighbors': [1, 5, 9]}

    #knn = GridSearchCV(knn, parameters_knn)

    return knn
