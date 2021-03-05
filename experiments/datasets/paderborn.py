"""
Class definition of Paderborn Bearing dataset download and acquisitions extraction.
"""

import urllib.request
import scipy.io
import numpy as np
import os
from sklearn.model_selection import KFold, StratifiedKFold

# Unpack Tools
from pyunpack import Archive


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
      Download raw compressed files from PADERBORN website
    load_acquisitions()
      Extract data from files
    """
    def __init__(self, debug=0):
        self.rawfilesdir = "paderborn_raw"
        self.dirdest = "paderborn_seg"
        self.url = "http://groups.uni-paderborn.de/kat/BearingDataCenter/"
        self.debug = debug
        if debug == 0:
            self.conditions = {"N": "normal",
                               "I": "inner",
                               "O": "outer"}
        else:
            self.conditions = {"N": [(20, 256000)],
                               "O": [(20, 256000)],
                               "I": [(20, 256000)]}

        self.n_folds = 5
        self.sample_size = 8192
        self.n_acquisitions = 20

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

        if debug == 0:
            normal_folder = ["K001", "K002", "K003", "K004", "K005", "K006"]
            OR_folder = ["KA01", "KA03", "KA04", "KA05", "KA06", "KA07", "KA08",
                        "KA09", "KA15", "KA16", "KA22", "KA30"]
            IR_folder = ["KI01", "KI03", "KI05", "KI07", "KI08", "KI16", "KI17",
                        "KI18", "KI21"]
            MIX_folder = ["KB23", "KB24", "KB27", "KI14"]  # VERIFICAR

        else:
            normal_folder = ["K002"]
            OR_folder = ["KA01"]
            IR_folder = ["KI01"]

        settings_files = ["N15_M07_F10_", "N09_M07_F10_", "N15_M01_F10_", "N15_M07_F04_"]

        if debug == 0:
            n = 20
        else:
            n = 5

        # Normal
        for folder in normal_folder:
            for idx, setting in enumerate(settings_files):
                for i in range(1, n + 1):
                    key = "Normal_" + folder + "_" + str(idx) + "_" + str(i)
                    files_path[key] = os.path.join(self.rawfilesdir, folder, setting + folder +
                                                   "_" + str(i) + ".mat")

        # OR
        for folder in OR_folder:
            for idx, setting in enumerate(settings_files):
                for i in range(1, n + 1):
                    key = "OR_" + folder + "_" + str(idx) + "_" + str(i)
                    files_path[key] = os.path.join(self.rawfilesdir, folder, setting + folder +
                                                   "_" + str(i) + ".mat")

        # IR
        for folder in IR_folder:
            for idx, setting in enumerate(settings_files):
                for i in range(1, n + 1):
                    key = "IR_" + folder + "_" + str(idx) + "_" + str(i)
                    files_path[key] = os.path.join(self.rawfilesdir, folder, setting + folder +
                                                   "_" + str(i) + ".mat")

        self.files = files_path
        print(self.files)

    def download(self):
        """
        Download and extract compressed files from Paderborn website.
        """

        # RAR Files names
        if self.debug == 0:
            rar_files_name = ["K001.rar", "K002.rar", "K003.rar", "K004.rar", "K005.rar", "K006.rar",
                              "KA01.rar", "KA03.rar", "KA04.rar", "KA05.rar", "KA06.rar", "KA07.rar",
                              "KA08.rar", "KA09.rar", "KA15.rar", "KA16.rar", "KA22.rar", "KA30.rar",
                              "KB23.rar", "KB24.rar", "KB27.rar",
                              "KI01.rar", "KI03.rar", "KI04.rar", "KI05.rar", "KI07.rar", "KI08.rar",
                              "KI14.rar", "KI16.rar", "KI17.rar", "KI18.rar", "KI21.rar"]
        else:
            rar_files_name = ["K002.rar", "KA01.rar", "KI01.rar"]

        url = self.url

        dirname = self.rawfilesdir
        dir_rar = "rar_files"
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
            if not os.path.isdir(os.path.join(dirname, dir_rar)):
                os.mkdir(os.path.join(dirname, dir_rar))
    
                print("Downloading RAR files:")
                for i in rar_files_name:
                    file_name = i
                    if not os.path.exists(os.path.join(dirname, dir_rar, file_name)):
                        urllib.request.urlretrieve(url + file_name, os.path.join(dirname, dir_rar, file_name))
                    print(file_name)
    
            print("Extracting files:")
            for i in rar_files_name:
                if not os.path.exists(os.path.join(dirname, i[:4])):
                    file_name = os.path.join(dirname, dir_rar, i)
                    Archive(file_name).extractall(dirname)
                    print(i)

    def load_acquisitions(self):
        """
        Extracts the acquisitions of each file in the dictionary files_names.
        """

        for key in self.files:
            if key != 'OR_KA08_2_2':
                matlab_file = scipy.io.loadmat(self.files[key])
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

        for train_index, test_index in kf.split(X):
            #print("Train Index: ", train_index, "Test Index: ", test_index)
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]

            yield X_train, y_train, X_test, y_test

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

        for train_index, test_index in kf.split(X, y):
            #print("Train Index: ", train_index, "Test Index: ", test_index)
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]

            yield X_train, y_train, X_test, y_test

    def groupkfold_custom(self):

      # Define folds index by samples
      samples_index = [0]
      final_sample = 0
      for condition in self.conditions.items():
        for acquisitions_details in condition[1]:
          samples_acquisition = acquisitions_details[1] // self.sample_size
          n_samples = acquisitions_details[0] * samples_acquisition
          fold_size_groups = acquisitions_details[0] // self.n_folds
          fold_size = fold_size_groups * samples_acquisition
          for i in range(self.n_folds - 1):
            samples_index.append(samples_index[-1] + fold_size)
          final_sample = final_sample + n_samples
          samples_index.append(final_sample)

      #print(samples_index)

      # Define folds split
      folds_split = []
      for i in range(self.n_folds):
        splits = [0] * self.n_folds
        splits[i] = 1
        folds_split.append(splits)

      # print(folds_split)

      folds = []
      for split in folds_split:
        fold_dict = {}
        for k in range(len(samples_index) - 1):
          pos = k % self.n_folds
          if split[pos] == 1:
            fold_dict[(samples_index[k], samples_index[k + 1])] = "test"
          else:
            fold_dict[(samples_index[k], samples_index[k + 1])] = "train"
        folds.append(fold_dict)

      # print(folds)

      # Yield folds
      for f in folds:
        print("Folds by samples index: ", f)
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
