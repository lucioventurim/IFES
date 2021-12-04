
from utils.persist_results import load_results
from statistics import mean, stdev
from sklearn.metrics import accuracy_score, f1_score


def scores(file):

    results = load_results(file)

    n_folds = 0
    for fold in results:
        current_fold = fold[3]
        if current_fold > n_folds:
            n_folds = current_fold

    accuracy = []
    f1_macro = []
    for fold in results:
        current_fold = fold[3]
        if current_fold == n_folds:
            accuracy.append(accuracy_score(fold[4], fold[5]))
            f1_macro.append(f1_score(fold[4], fold[5], average='macro'))
            print("## Splitting Strategy: ", fold[1], "##")
            print("# Classification Model: ", fold[2], "#")
            print("Accuracy: ", accuracy, "Mean: ", mean(accuracy), "Std: ", stdev(accuracy))
            print("F1 Macro: ", f1_macro, "Mean: ", mean(f1_macro), "Std: ", stdev(f1_macro))

            #print("Accuracy Mean / Std / F1 Macro Mean / Std:")
            #print(str(mean(accuracy)).replace('.',','))
            #print(str(stdev(accuracy)).replace('.',','))
            #print(str(mean(f1_macro)).replace('.',','))
            #print(str(stdev(f1_macro)).replace('.',','))

            print()
            accuracy = []
            f1_macro = []
        else:
            accuracy.append(accuracy_score(fold[4], fold[5]))
            f1_macro.append(f1_score(fold[4], fold[5], average='macro'))
