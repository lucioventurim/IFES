"""
Class definition of Paderborn Bearing dataset download and acquisitions extraction.
"""

import urllib.request
import scipy.io
import numpy as np
import os
from sklearn.model_selection import KFold, StratifiedKFold
import csv
import urllib
import rarfile
import shutil

# Unpack Tools
from pyunpack import Archive


def get_paderborn_bearings(file_name):
    # Get bearings to be considered to be
    cwd = os.getcwd()
    bearing_file = os.path.join(cwd, "experiments/datasets", file_name)

    bearing_names = []
    with open(bearing_file, 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            bearing_names = np.append(bearing_names, row)

    return bearing_names


def download_file(url, dirname, dir_rar, bearings):

    for i in bearings:
        print("Downloading Bearing Data:", i)
        file_name = i + ".rar"

        req = urllib.request.Request(url + file_name, method='HEAD')
        f = urllib.request.urlopen(req)
        file_size = int(f.headers['Content-Length'])

        dir_path = os.path.join(dirname, dir_rar, file_name)
        if not os.path.exists(dir_path):
            urllib.request.urlretrieve(url + file_name, dir_path)
            downloaded_file_size = os.stat(dir_path).st_size
        else:
            downloaded_file_size = os.stat(dir_path).st_size

        if file_size != downloaded_file_size:
            os.remove(dir_path)
            print("File Size Incorrect. Downloading Again.")
            download_file(url, dirname, dir_rar, bearings)


def extract_rar(bearings, dirname, dir_rar):

    for i in bearings:
        print("Extracting Bearing Data:", i)
        dir_bearing_rar = os.path.join(dirname, dir_rar, i + ".rar")
        dir_bearing_data = os.path.join(dirname, i)
        if not os.path.exists(dir_bearing_data):
            file_name = dir_bearing_rar
            Archive(file_name).extractall(dirname)
            extracted_files_qnt = len([name for name in os.listdir(dir_bearing_data)
                                       if os.path.isfile(os.path.join(dir_bearing_data, name))])
        else:
            extracted_files_qnt = len([name for name in os.listdir(dir_bearing_data)
                       if os.path.isfile(os.path.join(dir_bearing_data, name))])
        rf = rarfile.RarFile(dir_bearing_rar)
        rar_files_qnt = len(rf.namelist())

        if rar_files_qnt != extracted_files_qnt + 1:
            shutil.rmtree(dir_bearing_data)
            print("Extracted Files Incorrect. Extracting Again.")
            extract_rar(bearings, dirname, dir_rar)


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


class Paderborn():
    """
    Paderborn class wrapper for database download and acquisition.

    ...
    Attributes
    ----------
    rawfilesdir : str
      directory name where the files will be downloaded
    url : str
      website from the raw files are downloaded
    conditions : dict
      the keys represent the condition code and the values the number of acquisitions and its lengh
    files : dict
      the keys represent the conditions_acquisition and the values are the files names

    Methods
    -------
    download()
      Download and extract raw compressed files from PADERBORN website
    load_acquisitions()
      Extract vibration data from files
    """
    def __init__(self, debug=0):
        self.rawfilesdir = "paderborn_raw"
        self.url = "http://groups.uni-paderborn.de/kat/BearingDataCenter/"
        self.debug = debug
        self.n_folds = 3
        self.sample_size = 8192

        if debug == 0:
            self.bearing_names = get_paderborn_bearings("paderborn_bearings.csv")
            self.n_acquisitions = 20
        else:
            self.bearing_names = get_paderborn_bearings("paderborn_bearings_debug.csv")
            self.n_acquisitions = 3

        """
        Associate each file name to a bearing condition in a Python dictionary. 
        The dictionary keys identify the conditions.
    
        In total, experiments with 32 different bearings were performed:
        12 bearings with artificial damages and 14 bearings with damages
        from accelerated lifetime tests. Moreover, experiments with 6 healthy
        bearings and a different time of operation were performed as
        reference states.
    
        The rotational speed of the drive system, the radial force onto the test
        bearing and the load torque in the drive train are the main operation
        parameters. To ensure comparability of the experiments, fixed levels were
        defined for each parameter. All three parameters were kept constant for
        the time of each measurement. At the basic setup (Set no. 0) of the 
        operation parameters, the test rig runs at n = 1,500 rpm with a load 
        torque of M = 0.7 Nm and a radial force on the bearing of F = 1,000 N. Three
        additional settings are used by reducing the parameters one
        by one to n = 900 rpm, M = 0.1 Nm and F = 400 N (set No. 1-3), respectively.
    
        For each of the settings, 20 measurements of 4 seconds each were recorded
        for each bearing. There are a total of 2.560 files.
    
        All files start with the bearing code, followed by the conditions, by an
        algarism representing the setting and end with an algarism representing 
        the sample sequential. All features are separated by an underscore character.
        """
        files_path = {}

        settings_files = ["N15_M07_F10_", "N09_M07_F10_", "N15_M01_F10_", "N15_M07_F04_"]

        normal_qnt = 0
        OR_qnt = 0
        IR_qnt = 0

        for bearing in self.bearing_names:
            if bearing[1] == '0':
                tp = "Normal_"
                normal_qnt = normal_qnt + 1
            elif bearing[1] == 'A':
                tp = "OR_"
                OR_qnt = OR_qnt + 1
            else:
                tp = "IR_"
                IR_qnt = IR_qnt + 1
            for idx, setting in enumerate(settings_files):
                for i in range(1, self.n_acquisitions + 1):
                    key = tp + bearing + "_" + str(idx) + "_" + str(i)
                    files_path[key] = os.path.join(self.rawfilesdir, bearing, setting + bearing +
                                                   "_" + str(i) + ".mat")

        # Define number of acquisitions for each condition and its length
        self.conditions = {"N": [((self.n_acquisitions*normal_qnt*len(settings_files)), 256000)],
                           "O": [((self.n_acquisitions*OR_qnt*len(settings_files)), 256000)],
                           "I": [((self.n_acquisitions*IR_qnt*len(settings_files)), 256000)]
                           }

        self.files = files_path
        #print(len(self.files))

    def download(self):
        """
        Download and extract compressed files from Paderborn website.
        """

        # Download RAR Files
        url = self.url
        dirname = self.rawfilesdir
        dir_rar = "rar_files"
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        if not os.path.isdir(os.path.join(dirname, dir_rar)):
            os.mkdir(os.path.join(dirname, dir_rar))

        print("Downloading RAR files:")
        download_file(url, dirname, dir_rar, self.bearing_names)

        print("Extracting files:")
        extract_rar(self.bearing_names, dirname, dir_rar)

    def load_acquisitions(self):
        """
        Extracts the acquisitions of each file in the dictionary files_names.
        """

        cwd = os.getcwd()
        for key in self.files:
            if key != 'OR_KA08_2_2':
                matlab_file = scipy.io.loadmat(os.path.join(cwd, self.files[key]))
                if len(self.files[key]) > 41:
                    vibration_data = matlab_file[self.files[key][19:38]]['Y'][0][0][0][6][2]
                else:
                    vibration_data = matlab_file[self.files[key][19:37]]['Y'][0][0][0][6][2]
            #print(len(vibration_data[0]))
            yield key, vibration_data[0]

    def kfold(self):
        X = np.empty((0,self.sample_size))
        y = []

        for key, acquisition in self.load_acquisitions():
            acquisition_size = len(acquisition)
            samples_acquisition = acquisition_size // self.sample_size
            for i in range(samples_acquisition):
                sample = acquisition[(i * self.sample_size):((i + 1) * self.sample_size)]
                X = np.append(X, np.array([sample]), axis=0)
                y = np.append(y, key[0])

        kf = KFold(n_splits=self.n_folds)

        for train, test in kf.split(X):
            # print("Train Index: ", train, "Test Index: ", test)
            yield X[train], y[train], X[test], y[test]

    def stratifiedkfold(self):

        X = np.empty((0,self.sample_size))
        y = []

        for key, acquisition in self.load_acquisitions():
            acquisition_size = len(acquisition)
            samples_acquisition = acquisition_size // self.sample_size
            for i in range(samples_acquisition):
                sample = acquisition[(i * self.sample_size):((i + 1) * self.sample_size)]
                X = np.append(X, np.array([sample]), axis=0)
                y = np.append(y, key[0])

        kf = StratifiedKFold(n_splits=self.n_folds)

        for train, test in kf.split(X, y):
            # print("Train Index: ", train, "Test Index: ", test)
            yield X[train], y[train], X[test], y[test]

    def groupkfold_custom(self):

        # Define folds index by samples sequential
        samples_index = group_folds_index(self.conditions, self.sample_size, self.n_folds)
        #print(samples_index)

        # Define folds split
        folds = group_folds_split(self.n_folds, samples_index)
        # print(folds)

        # Yield folds
        for f in folds:
            #print("Folds by samples index: ", f)
            X_train = []
            y_train = []
            X_test = []
            y_test = []

            counter = 0
            for key, acquisition in self.load_acquisitions():
                acquisition_size = len(acquisition)
                samples_acquisition_fold = acquisition_size // self.sample_size
                for i in range(samples_acquisition_fold):
                    sample = acquisition[(i * self.sample_size):((i + 1) * self.sample_size)]
                    res = ""
                    for (k1, k2) in f:
                        if (k1 <= counter and k2 > counter):
                            res = f[(k1, k2)]
                    if res == "train":
                        X_train.append(sample)
                        y_train.append(key[0])
                    else:
                        X_test.append(sample)
                        y_test.append(key[0])
                    counter = counter + 1

            yield X_train, y_train, X_test, y_test
