

from classification_models import auto_knn, auto_random_forest
#from classification_models import auto_cnn, auto_alexnet
from sklearn.metrics import accuracy_score, f1_score
from statistics import mean, stdev

from datasets.mfpt import MFPT

def main():
    #debug = 0
    mfpt = MFPT()
    mfpt.download()

    clfs = [#('CNN', auto_cnn.instantiate_auto_cnn()),
            #('AlexNet', auto_alexnet.instantiate_auto_alexnet()),
            ('K-Nearest Neighbors', auto_knn.instantiate_auto_knn()),
            ('Random Forest', auto_random_forest.instantiate_auto_random_forest())]

    for clf in clfs:
        accuracy = []
        f1_macro = []
        print(clf[0])
        for X_train, y_train, X_test, y_test in mfpt.mfpt_kfold():
            clf[1].fit(X_train, y_train)
            y_pred = clf[1].predict(X_test)
            accuracy.append(accuracy_score(y_test, y_pred))
            f1_macro.append(f1_score(y_test, y_pred, average='macro'))
        print("Accuracy: ", accuracy, "Mean: ", mean(accuracy), "Std: ", stdev(accuracy))
        print("F1 Macro: ", f1_macro, "Mean: ", mean(f1_macro), "Std: ", stdev(f1_macro))


if __name__ == "__main__":
    main()
