# Segmentation

import numpy as np


def segmentator(acquisitions, sample_size, debug):

    n = len(acquisitions)

    X = np.empty((0, sample_size))
    y = []
    groups = []

    for i, key in enumerate(acquisitions):
        acquisition_size = len(acquisitions[key])
        if debug:
            n_samples = 15
        else:
            n_samples = acquisition_size // sample_size
        print('{}/{} --- {}: {}'.format(i + 1, n, key, n_samples))
        X = np.concatenate((X, acquisitions[key][:(n_samples * sample_size)].reshape((n_samples, sample_size))))

        for j in range(n_samples):
            groups.append(key)
            y.append(key[0])

    return X, y, groups
