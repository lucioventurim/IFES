
from classification_models import auto_knn, auto_random_forest
#from classification_models import auto_cnn
from utils import persist_results, metrics
import os

from datasets.mfpt import MFPT
from datasets.paderborn import Paderborn

import time
import functools
def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        with open("execution_time", 'a') as file:
            file.write(f"{run_time} s\n")
        return value
    return wrapper_timer

@timer
def experimenter(dataset, clfs, splits):
    print("### Dataset: ", dataset[0], "###")
    with open("execution_time", 'a') as file:
        file.write(f"{dataset[0]}: ")
    dataset[1].download()
    results = []
    print("Performing Experiments.")
    for folds in splits:
        for clf in clfs:
            fold_number = 1
            for X_train, y_train, X_test, y_test in getattr(dataset[1], folds[1])():
                print(folds[0], clf[0])
                #clf[1].fit(X_train, y_train)
                #y_pred = clf[1].predict(X_test)
                #y_proba = clf[1].predict_proba(X_test)
                #results.append([dataset[0], folds[0], clf[0], fold_number, y_test, y_pred, y_proba])
                #fold_number = fold_number + 1
    #saved_results = persist_results.save_results(results)
    #metrics.scores(saved_results)


def main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    clfs = [('K-Nearest Neighbors', auto_knn.instantiate_auto_knn()),
            #('Random Forest', auto_random_forest.instantiate_auto_random_forest())
            # ('CNN', auto_cnn.instantiate_auto_cnn()),
            # ('AlexNet', auto_alexnet.instantiate_auto_alexnet())
            ]

    splits = [#('Kfold', 'kfold'),
              #('StratifiedKfold', 'stratifiedkfold'),
              #('GroupKfold by Acquisition', 'groupkfold_acquisition'),
              ('GroupKfold by Load', 'groupkfold_load'),
             ]

    #dataset = ('MFPT', MFPT())
    #experimenter(dataset, clfs, splits)


    dataset = ('Paderborn3', Paderborn(bearing_names_file="paderborn_bearings_debug.csv", n_aquisitions=4))
    experimenter(dataset, clfs, splits)
    
    #dataset = ('Paderborn20', Paderborn(bearing_names_file="paderborn_bearings_debug.csv"))
    #experimenter(dataset, clfs, splits)


if __name__ == "__main__":
    main()
