
from datetime import datetime
import csv
import ast


def save_results(results):

    now = datetime.now()
    date_time = now.strftime("%Y.%m.%d_%H.%M.%S")
    result_file_name = date_time + ".csv"

    with open(result_file_name, 'w', newline="") as csvfile:
        fieldnames = ['dataset', 'split', 'classifier', 'fold', 'y_actual', 'y_pred', 'y_proba']
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(results)

    return result_file_name


def load_results(file):

    results = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                pass
            else:
                dataset = row[0]
                split = row[1]
                classifier = row[2]
                fold = int(row[3])
                y_actual = ast.literal_eval(row[4])
                if len(y_actual) == 1:
                    y_actual = list(y_actual[0])
                y_pred = ast.literal_eval(row[5])
                if len(y_pred) == 1:
                    y_pred = list(y_pred[0])
                y_proba = row[6]
                row_results = [dataset, split, classifier, fold, y_actual, y_pred, y_proba]
                results.append(row_results)
                line_count += 1

    return results
