

from datetime import datetime
import csv

def save_results(results):

    now = datetime.now()
    date_time = now.strftime("%Y.%m.%d_%H.%M.%S")
    result_file_name = date_time + ".csv"

    with open(result_file_name, 'w') as f:
        for key in results.keys():
            f.write("%s,%s\n" % (key, results[key]))

    return result_file_name

def load_results(file):

    #a_csv_file = open(file, "r")
    #dict_reader = csv.DictReader(a_csv_file)

    #ordered_dict_from_csv = list(dict_reader)[0]
    #dict_from_csv = dict(ordered_dict_from_csv)

    print(file)
