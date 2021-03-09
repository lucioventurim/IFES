

from classification_models import auto_knn, auto_random_forest
#from classification_models import auto_cnn, auto_alexnet
from utils import persist_results, metrics

from datasets.mfpt import MFPT
from datasets.paderborn import Paderborn


def experimenter(datasets, clfs, splits):

    for data in datasets:
        print("### Dataset: ", data[0], "###")
        data[1].download()

        results = []
        for folds in splits:
            for clf in clfs:
                fold_number = 1
                for X_train, y_train, X_test, y_test in getattr(data[1], folds[1])():
                    clf[1].fit(X_train, y_train)
                    y_pred = clf[1].predict(X_test)
                    y_proba = clf[1].predict_proba(X_test)
                    results.append([data[0], folds[0], clf[0], fold_number, y_test, y_pred, y_proba])
                    fold_number = fold_number + 1

        saved_results = persist_results.save_results(results)
        metrics.scores(saved_results)

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

    experimenter(datasets, clfs, splits)


if __name__ == "__main__":
    main()
