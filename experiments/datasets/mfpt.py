"""
Class definition of MFPT Bearing dataset download and acquisitions extraction.
"""

import urllib.request
import scipy.io
import numpy as np
import os
from sklearn.model_selection import KFold, StratifiedKFold

# Unpack Tools
from pyunpack import Archive


class MFPT():
  """
  MFPT class wrapper for database download and acquisition.

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
    Download raw compressed files from MFPT website
  load_acquisitions()
    Extract data from files
  """
  def __init__(self, debug = 0):
    self.rawfilesdir = "mfpt_raw"
    self.url="https://mfpt.org/wp-content/uploads/2020/02/MFPT-Fault-Data-Sets-20200227T131140Z-001.zip"
    self.conditions = {"N": [(3, 585936)],
              "O": [(3, 585936), (7, 146484)],
              "I": [(7, 146484)]}
    self.debug = debug

    self.n_folds = 3
    self.sample_size = 8192
    self.n_acquisitions = 20


    """
    The MFPT dataset is divided into 3 kinds of states: normal state, inner race
    fault state, and outer race fault state (N, IR, and OR), where three baseline
    data and three outer race fault were gathered at a sampling frequency of 97656 Hz and under 270 lbs of
    load for 6 seconds; seven outer race fault data were gathered at a sampling frequency of
    48828 Hz and, respectively, under 25, 50, 100, 150, 200, 250, and 300 lbs 
    of load, and seven inner race fault data were gathered at a sampling 
    frequency of 48828 Hz and, respectively, under 0, 50, 100, 150, 200, 250, 
    and 300 lbs of load, all for 3 seconds.
    """
    files_path = {}

    # Normal
    files_path["Normal_0"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/1 - Three Baseline Conditions/baseline_1")
    files_path["Normal_1"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/1 - Three Baseline Conditions/baseline_2")
    files_path["Normal_2"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/1 - Three Baseline Conditions/baseline_3")
    # OR
    files_path["OR_0"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/2 - Three Outer Race Fault Conditions/OuterRaceFault_1")
    files_path["OR_1"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/2 - Three Outer Race Fault Conditions/OuterRaceFault_2")
    files_path["OR_2"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/2 - Three Outer Race Fault Conditions/OuterRaceFault_3")
    files_path["OR_3"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_1")
    files_path["OR_4"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_2")
    files_path["OR_5"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_3")
    files_path["OR_6"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_4")
    files_path["OR_7"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_5")
    files_path["OR_8"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_6")
    files_path["OR_9"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_7")
    # IR
    files_path["IR_0"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_1")
    files_path["IR_1"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_2")
    files_path["IR_2"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_3")
    files_path["IR_3"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_4")
    files_path["IR_4"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_5")
    files_path["IR_5"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_6")
    files_path["IR_6"] = os.path.join(self.rawfilesdir, "MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_7")

    self.files = files_path

  def download(self):
    """
    Download and extract compressed files from MFPT website.
    """

    url = self.url

    dirname = self.rawfilesdir
    if not os.path.isdir(dirname):
      os.mkdir(dirname)

      zip_name = "MFPT-Fault-Data-Sets-20200227T131140Z-001.zip"

      print("Downloading ZIP file")

      urllib.request.urlretrieve(url, os.path.join(dirname, zip_name))

      print("Extracting files")
      file_name = os.path.join(dirname, zip_name)
      Archive(file_name).extractall(dirname)

  def load_acquisitions(self):
    """
    Extracts the acquisitions of each file in the dictionary files_names.
    """

    for key in self.files:
      matlab_file = scipy.io.loadmat(self.files[key])

      if len(key) == 8:
        vibration_data_raw = matlab_file['bearing'][0][0][1]
      else:
        vibration_data_raw = matlab_file['bearing'][0][0][2]

      vibration_data = np.array([ elem for singleList in vibration_data_raw for elem in singleList])

      #print(len(vibration_data))
      yield key, vibration_data

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

    #print(len(X))
    kf = KFold(n_splits=3)

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

    #print(X)
    kf = StratifiedKFold(n_splits=3)

    for train, test in kf.split(X, y):
      #print("Train Index: ", train, "Test Index: ", test)
      yield X[train], y[train], X[test], y[test]

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
