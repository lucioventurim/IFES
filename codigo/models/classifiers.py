# Classifiers and Scoring Definitions

from models.auto_alexnet import instantiate_auto_alexnet
from models.auto_knn import instantiate_auto_knn
from models.auto_random_forest import instantiate_auto_random_forest
from models.auto_cnn import instantiate_auto_cnn


def Classifiers():

    # Chosen Classifiers

    clfs = [#('CNN', instantiate_auto_cnn()),
            ('K-Nearest Neighbors', instantiate_auto_knn()),
            #('AlexNet', instantiate_auto_alexnet()),
            ('Random Forest', instantiate_auto_random_forest())]

    return clfs


def Scoring():

    scoring = ['accuracy', 'f1_macro']

    return scoring
