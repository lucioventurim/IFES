

from classification_models import auto_knn, auto_random_forest
#from classification_models import auto_cnn, auto_alexnet
from sklearn.metrics import accuracy_score, f1_score
from statistics import mean, stdev
from utils import persist_results

import numpy as np

from datasets.mfpt import MFPT
from datasets.paderborn import Paderborn

def main():

    debug = 1

    datasets = [#('MFPT', MFPT(debug=debug)),
                ('Paderborn', Paderborn(debug=debug)),
                ]

    clfs = [('K-Nearest Neighbors', auto_knn.instantiate_auto_knn()),
            ('Random Forest', auto_random_forest.instantiate_auto_random_forest())
            # ('CNN', auto_cnn.instantiate_auto_cnn()),
            # ('AlexNet', auto_alexnet.instantiate_auto_alexnet())
            ]

    splits = [#('Kfold', 'kfold'),
              ('StratifiedKfold', 'stratifiedkfold'),
              ('GroupKfold', 'groupkfold_custom'),
             ]

    for data in datasets:
        print("### ", data[0], " ###")

        data[1].download()

        results = {}
        for folds in splits:
            print("### ", folds[0], " ###")
            for clf in clfs:
                accuracy = []
                f1_macro = []
                print(clf[0])
                fold_number = 1
                for X_train, y_train, X_test, y_test in getattr(data[1], folds[1])():
                    clf[1].fit(X_train, y_train)
                    y_pred = clf[1].predict(X_test)
                    y_proba = clf[1].predict_proba(X_test)
                    accuracy.append(accuracy_score(y_test, y_pred))
                    f1_macro.append(f1_score(y_test, y_pred, average='macro'))
                    results[folds[0] + "_" + clf[0] + "_" + str(fold_number)] = [np.array(y_test), y_pred, y_proba]
                    fold_number = fold_number + 1
                print("Accuracy: ", accuracy, "Mean: ", mean(accuracy), "Std: ", stdev(accuracy))
                print("F1 Macro: ", f1_macro, "Mean: ", mean(f1_macro), "Std: ", stdev(f1_macro))

        saved_results = persist_results.save_results(results)
        persist_results.load_results(saved_results)


if __name__ == "__main__":
    main()
