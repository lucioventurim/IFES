# Performer

import numpy as np
from split.kfold import split_kfold
from split.groupkfold import split_groupkfold


def perfomer(clfs, X, y, groups, scoring, verbose):

    # Estimators
    for clf_name, estimator in clfs:
        print("*" * (len(clf_name) + 8), '\n***', clf_name, '***\n' + "*" * (len(clf_name) + 8))

        split_kfold(estimator, X, np.array(y), scoring, verbose, clf_name)

        split_groupkfold(estimator, X, np.array(y), groups, scoring, verbose)
