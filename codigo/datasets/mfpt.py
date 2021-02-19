"""
Class definition of MFPT Bearing dataset download and acquisitions extraction.
"""

import urllib.request
from database import Database_Download
import scipy.io
import numpy as np
import pickle
import os


# Unpack Tools
from pyunpack import Archive


class MFPT(Database_Download):
  """
  MFPT class wrapper for database download and acquisition.

  ...
  Attributes
  ----------
  rawfilesdir : str
    directory name where the files will be downloaded
  dirdest : str
    directory name of the segmented files
  url : str
    website from the raw files are downloaded
  conditions : dict
    the keys represent the condition code and the values the condition name
  files : dict
    the keys represent the conditions_acquisition and the values are the files names

  Methods
  -------
  download()
    Download raw compressed files from MFPT website
  acquisitions()
    Extract data from files
  load()
    Load acquisitions previsously saved in picke file
  """
  def __init__(self, debug = 0):
    self.rawfilesdir = "mfpt_raw"
    self.dirdest = "mfpt_seg"
    self.url="https://mfpt.org/wp-content/uploads/2020/02/MFPT-Fault-Data-Sets-20200227T131140Z-001.zip"
    self.conditions = {"N":"normal",
              "I": "inner",
              "O": "outer"}
    self.debug = debug

    """
    The MFPT dataset is divided into 3 kinds of states: normal state, inner race
    fault state, and outer race fault state (N, IR, and OR), where three baseline
    data were gathered at a sampling frequency of 97656 Hz and under 270 lbs of
    load; seven outer race fault data were gathered at a sampling frequency of
    48828 Hz and, respectively, under 25, 50, 100, 150, 200, 250, and 300 lbs 
    of load, and seven inner race fault data were gathered at a sampling 
    frequency of 48828 Hz and, respectively, under 0, 50, 100, 150, 200, 250, 
    and 300 lbs of load.
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

  def acquisitions(self):
    """
    Extracts the acquisitions of each file in the dictionary files_names.
    The user must be careful because this function converts all files
    in the files_names in numpy arrays.
    As large the number of entries in files_names
    as large will be the space of memory necessary.

    Returns
    -------
    acquisitions_data : dict
    Returns the sample rate, the sample size, the conditions dict, the destinations directory
    and the acquisitions dict, where the keys represent the bearing code, followed by the conditions, by an
    algarism representing the setting and end with an algarism representing
    the sample sequential. All features are separated by an underscore character.
    """

    acquisitions_dict = {}
    for key in self.files:
      matlab_file = scipy.io.loadmat(self.files[key])

      if len(key) == 8:
        vibration_data_raw = matlab_file['bearing'][0][0][1]
      else:
        vibration_data_raw = matlab_file['bearing'][0][0][2]

      vibration_data = np.array([ elem for singleList in vibration_data_raw for elem in singleList])

      acquisitions_dict[key] = vibration_data

    acquisitions_data = {}
    acquisitions_data['conditions'] = self.conditions
    acquisitions_data['dirdest'] = self.dirdest
    acquisitions_data['acquisitions'] = acquisitions_dict

    return acquisitions_data

  def load(self):
    """
    Load the data set.

    Returns
    -------
    acquisitions : dict
    Returns the sample rate, the sample size, the conditions dict, the destinations directory
    and the acquisitions dict, where the keys represent the bearing code, followed by the conditions, by an
    algarism representing the setting and end with an algarism representing
    the sample sequential. All features are separated by an underscore character.
    """

    pickle_file = 'mfpt.pickle'

    if os.path.isfile(pickle_file):
      with open(pickle_file, 'rb') as handle:
        acquisitions = pickle.load(handle)
    else:
      self.download()
      acquisitions = self.acquisitions()
      with open(pickle_file, 'wb') as handle:
        pickle.dump(acquisitions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return acquisitions
