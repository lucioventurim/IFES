# Classifiers and Scoring Definitions

from models.alexnet import clf_alexnet
from models.knn import clf_knn
from models.randomforest import clf_rf


def Classifiers():

    # Chosen Classifiers

    clfs = [('K-Nearest Neighbors', clf_knn()), ('AlexNet', clf_alexnet()), ('Random Forest', clf_rf())]

    return clfs


def Scoring():

    scoring = ['accuracy', 'f1_macro']

    return scoring
