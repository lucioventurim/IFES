# Split - KFold

from sklearn.model_selection import cross_validate


def split_kfold(estimator, X, y, scoring, verbose):

    print("---Kfold---")
    estimator_kfold = estimator

    score = cross_validate(estimator_kfold, X, y, scoring=scoring, cv=4, verbose=verbose)
    for metric, s in score.items():
        print(metric, ' \t', s, ' Mean: ', format(s.mean(), '.2f'), ' Std: ', format(s.std(), '.2f'))
