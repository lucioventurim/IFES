

import pickle
from datetime import datetime


def save_pickle(results):

    now = datetime.now()
    date_time = now.strftime("%Y.%m.%d_%H.%M.%S")
    result_file_name = date_time + ".pickle"

    with open(result_file_name, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return result_file_name

def load_pickle(file):

    with open(file, 'rb') as handle:
        b = pickle.load(handle)

    print(b)
