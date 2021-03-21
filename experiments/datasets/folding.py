

# Folding utils

def group_folds_index(conditions, sample_size, n_folds):
    # Define folds index by samples sequential for groups
    samples_index = [0]
    final_sample = 0
    for condition in conditions.items():
        for acquisitions_details in condition[1]:
            samples_acquisition = acquisitions_details[1] // sample_size
            n_samples = acquisitions_details[0] * samples_acquisition
            fold_size_groups = acquisitions_details[0] // n_folds
            fold_size = fold_size_groups * samples_acquisition
            for i in range(n_folds - 1):
                samples_index.append(samples_index[-1] + fold_size)
            final_sample = final_sample + n_samples
            samples_index.append(final_sample)

    return samples_index


def group_folds_split(n_folds, samples_index):
    # Define folds split for groups
    folds_split = []
    for i in range(n_folds):
        splits = [0] * n_folds
        splits[i] = 1
        folds_split.append(splits)

    # print(folds_split)

    folds = []
    for split in folds_split:
        fold_dict = {}
        for k in range(len(samples_index) - 1):
            pos = k % n_folds
            if split[pos] == 1:
                fold_dict[(samples_index[k], samples_index[k + 1])] = "test"
            else:
                fold_dict[(samples_index[k], samples_index[k + 1])] = "train"
        folds.append(fold_dict)

    return folds
