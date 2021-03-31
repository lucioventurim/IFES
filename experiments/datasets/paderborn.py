"""
Class definition of Paderborn Bearing dataset download and acquisitions extraction.
"""

import urllib.request
import scipy.io
import numpy as np
import os
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedShuffleSplit
import csv
import urllib
import rarfile
import shutil
import sys

# Unpack Tools
from pyunpack import Archive

# Code to avoid incomplete array results
np.set_printoptions(threshold=sys.maxsize)


def download_file(url, dirname, dir_rar, bearing):

    print("Downloading Bearing Data:", bearing)
    file_name = bearing + ".rar"

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
        download_file(url, dirname, dir_rar, bearing)


def extract_rar(dirname, dir_rar, bearing):

    print("Extracting Bearing Data:", bearing)
    dir_bearing_rar = os.path.join(dirname, dir_rar, bearing + ".rar")
    dir_bearing_data = os.path.join(dirname, bearing)
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
        extract_rar(dirname, dir_rar, bearing)


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

    def get_paderborn_bearings(self):
        # Get bearings to be considered to be

        bearing_file = os.path.join("datasets", self.bearing_names_file)

        bearing_names = []
        with open(bearing_file, 'r') as fd:
            reader = csv.reader(fd)
            for row in reader:
                bearing_names = np.append(bearing_names, row)

        return bearing_names

    def __init__(self, bearing_names_file="paderborn_bearings.csv", n_aquisitions=20):
        self.rawfilesdir = "paderborn_raw"
        self.url = "http://groups.uni-paderborn.de/kat/BearingDataCenter/"
        self.n_folds = 4
        self.sample_size = 8192
        self.n_samples_acquisition = 30
        self.bearing_names_file = bearing_names_file
        self.bearing_names = self.get_paderborn_bearings()
        self.n_acquisitions = n_aquisitions

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


        settings_files = ["N15_M07_F10_", "N09_M07_F10_", "N15_M01_F10_", "N15_M07_F04_"]

        normal_qnt = 0
        OR_qnt = 0
        IR_qnt = 0


        # Files Paths ordered by bearings
        files_path = {}

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

        # Files Paths ordered by settings
        files_path_settings = {}

        for idx, setting in enumerate(settings_files):
            for bearing in self.bearing_names:
                if bearing[1] == '0':
                    tp = "Normal_"
                elif bearing[1] == 'A':
                    tp = "OR_"
                else:
                    tp = "IR_"
                for i in range(1, self.n_acquisitions + 1):
                    key = tp + bearing + "_" + str(idx) + "_" + str(i)
                    files_path_settings[key] = os.path.join(self.rawfilesdir, bearing, setting + bearing +
                                                   "_" + str(i) + ".mat")

        # Define number of acquisitions for each condition and its length
        self.conditions = {"N": [((self.n_acquisitions*normal_qnt*len(settings_files)), 256000)],
                           "O": [((self.n_acquisitions*OR_qnt*len(settings_files)), 256000)],
                           "I": [((self.n_acquisitions*IR_qnt*len(settings_files)), 256000)]
                           }

        self.files = files_path
        self.files_settings = files_path_settings

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

        print("Downloading and Extracting RAR files:")

        for bearing in self.bearing_names:
            download_file(url, dirname, dir_rar, bearing)
            extract_rar(dirname, dir_rar, bearing)

        print("Dataset Loaded.")

    def load_acquisitions(self, files_path):
        """
        Extracts the acquisitions of each file in the dictionary files_names.
        """

        cwd = os.getcwd()
        for key in files_path:
            if key != 'OR_KA08_2_2':
                matlab_file = scipy.io.loadmat(os.path.join(cwd, files_path[key]))
                if len(files_path[key]) > 41:
                    vibration_data = matlab_file[files_path[key][19:38]]['Y'][0][0][0][6][2]
                else:
                    vibration_data = matlab_file[files_path[key][19:37]]['Y'][0][0][0][6][2]
            yield key, vibration_data[0]

    def kfold(self):
        X = np.empty((0,self.sample_size))
        y = []

        for key, acquisition in self.load_acquisitions(self.files):
            for i in range(self.n_samples_acquisition):
                sample = acquisition[(i * self.sample_size):((i + 1) * self.sample_size)]
                X = np.append(X, np.array([sample]), axis=0)
                y = np.append(y, key[0])

        kf = KFold(n_splits=self.n_folds, shuffle=True)

        for train, test in kf.split(X):
            # print("Train Index: ", train, "Test Index: ", test)
            yield X[train], y[train], X[test], y[test]

    def stratifiedkfold(self):

        X = np.empty((0,self.sample_size))
        y = []

        for key, acquisition in self.load_acquisitions(self.files):
            for i in range(self.n_samples_acquisition):
                sample = acquisition[(i * self.sample_size):((i + 1) * self.sample_size)]
                X = np.append(X, np.array([sample]), axis=0)
                y = np.append(y, key[0])

        kf = StratifiedShuffleSplit(n_splits=self.n_folds)

        for train, test in kf.split(X, y):
            # print("Train Index: ", train, "Test Index: ", test)
            yield X[train], y[train], X[test], y[test]

    def groupkfold_acquisition(self):

        X = np.empty((0, self.sample_size))
        y = []
        groups = []

        for key, acquisition in self.load_acquisitions(self.files):
            for i in range(self.n_samples_acquisition):
                sample = acquisition[(i * self.sample_size):((i + 1) * self.sample_size)]
                X = np.append(X, np.array([sample]), axis=0)
                y = np.append(y, key[0])
                groups = np.append(groups, int(key[-1]) % self.n_folds)

        kf = GroupKFold(n_splits=self.n_folds)

        for train, test in kf.split(X, y, groups):
            # print("Train Index: ", train, "Test Index: ", test)
            yield X[train], y[train], X[test], y[test]

    def groupkfold_settings(self):
        X = np.empty((0, self.sample_size))
        y = []

        for key, acquisition in self.load_acquisitions(self.files_settings):
            for i in range(self.n_samples_acquisition):
                sample = acquisition[(i * self.sample_size):((i + 1) * self.sample_size)]
                X = np.append(X, np.array([sample]), axis=0)
                y = np.append(y, key[0])

        kf = KFold(n_splits=self.n_folds)

        for train, test in kf.split(X):
            #print("Train Index: ", train, "Test Index: ", test)
            yield X[train], y[train], X[test], y[test]

    def groupkfold_bearings(self):
        X = np.empty((0, self.sample_size))
        y = []
        groups = []

        n_keys_bearings = 1
        n_normal = 1
        n_outer = 1
        n_inner = 1

        for key, acquisition in self.load_acquisitions(self.files):
            for i in range(self.n_samples_acquisition):
                sample = acquisition[(i * self.sample_size):((i + 1) * self.sample_size)]
                X = np.append(X, np.array([sample]), axis=0)
                y = np.append(y, key[0])
                if key[0] == 'N':
                    groups = np.append(groups, n_normal % self.n_folds)
                if key[0] == 'O':
                    groups = np.append(groups, n_outer % self.n_folds)
                if key[0] == 'I':
                    groups = np.append(groups, n_inner % self.n_folds)
            if n_keys_bearings < 4*self.n_acquisitions:
                n_keys_bearings = n_keys_bearings + 1
            elif key[0] == 'N':
                n_normal = n_normal + 1
                n_keys_bearings = 1
            elif key[0] == 'O':
                n_outer = n_outer + 1
                n_keys_bearings = 1
            elif key[0] == 'I':
                n_inner = n_inner + 1
                n_keys_bearings = 1

        kf = GroupKFold(n_splits=self.n_folds)

        for train, test in kf.split(X, y, groups):
            #print("Train Index: ", train, "Test Index: ", test)
            yield X[train], y[train], X[test], y[test]
