# Split - KFold

from sklearn.model_selection import cross_validate
import numpy as np
from datetime import datetime

def split_kfold(estimator, X, y, scoring, verbose, clf_name):

    print("---Kfold---")
    estimator_kfold = estimator

    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    result_file_name = clf_name.strip() + "_Kfold_" + date_time + ".csv"
    print(result_file_name)

    def get_preds(clf, XG, yg):
        with open(result_file_name, "ab+") as f:
            np.savetxt(f, ["y_actual: "], fmt="%s", newline=' ')
            np.savetxt(f, yg[np.newaxis], fmt="%s")
            np.savetxt(f, ["y_pred: "], fmt="%s", newline=' ')
            np.savetxt(f, clf.predict(XG)[np.newaxis], fmt="%s")
        return 0

    scoring = {'preds': get_preds,
               'accuracy': 'accuracy',
               'f1': 'f1_macro'}

    score = cross_validate(estimator_kfold, X, y,
                   scoring=scoring,
                   cv=3, verbose=verbose)

    #score = cross_validate(estimator_kfold, X, y, scoring=scoring, cv=3, verbose=verbose)
    for metric, s in score.items():
        print(metric, ' \t', s, ' Mean: ', format(s.mean(), '.2f'), ' Std: ', format(s.std(), '.2f'))
