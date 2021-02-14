# Split - Group KFold

from sklearn.model_selection import cross_validate, GroupKFold


def split_groupkfold(estimator, X, y, groups, scoring, verbose):

    print("---GroupKfold---")
    estimator_groupkfold = estimator

    score = cross_validate(estimator_groupkfold, X, y, groups, scoring, cv=GroupKFold(n_splits=4), verbose=verbose)

    for metric, s in score.items():
        print(metric, ' \t', s, ' Mean: ', format(s.mean(), '.2f'), ' Std: ', format(s.std(), '.2f'))
