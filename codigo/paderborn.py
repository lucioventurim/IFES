"""
Class definition of Paderborn Bearing dataset download and acquisitions extraction.
"""

import urllib.request
import database
import scipy.io
import pickle
import os

# Unpack Tools
from pyunpack import Archive

def files_debug(dirfiles):
  """
  Associate each Matlab file name to a bearing condition in a Python dictionary. 
  The dictionary keys identify the conditions.
  
  NOTE: Used only for debug. Use "debug=1" on intialization.
  """

  files_path = {}
  
  normal_folder = ["K002"]
  OR_folder = ["KA01"]
  IR_folder = ["KI01"]
  MIX_folder = ["KB23"] # VERIFICAR

  settings_files = ["N15_M07_F10_", "N09_M07_F10_", "N15_M01_F10_", "N15_M07_F04_"]

  n = 20

  # Normal
  for folder in normal_folder:
    for idx, setting in enumerate(settings_files):
      for i in range(1, n+1):
        key = "Normal_" + folder + "_" + str(idx) + "_" + str(i)
        files_path[key] = os.path.join(dirfiles, folder, setting + folder +
                                       "_" + str(i) + ".mat")

  # OR
  for folder in OR_folder:
    for idx, setting in enumerate(settings_files):
      for i in range(1, n+1):
        key = "OR_" + folder + "_" + str(idx) + "_" + str(i)
        files_path[key] = os.path.join(dirfiles, folder, setting + folder +
                                       "_" + str(i) + ".mat")

  # IR
  for folder in IR_folder:
    for idx, setting in enumerate(settings_files):
      for i in range(1, n+1):
        key = "IR_" + folder + "_" + str(idx) + "_" + str(i)
        files_path[key] = os.path.join(dirfiles, folder, setting + folder +
                                       "_" + str(i) + ".mat")

  return files_path


class Paderborn(database.Database_Download):
  """
  Paderborn class wrapper for database download and acquisition.

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
    Download raw compressed files from PADERBORN website
  acquisitions()
    Extract data from files
  load()
    Load acquisitions
  """

  def __init__(self, debug = 0):
    self.rawfilesdir = "paderborn_raw"
    self.dirdest = "paderborn_seg"
    self.url="http://groups.uni-paderborn.de/kat/BearingDataCenter/"
    self.conditions = {"N":"normal", 
                  "I": "inner", 
                  "O": "outer"}
    self.debug = debug

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
    
    #normal_folder = ["K001", "K002", "K003", "K004", "K005", "K006"]
    normal_folder = ["K001", "K002", "K003"]
    #OR_folder = ["KA01", "KA03", "KA04", "KA05", "KA06", "KA07", "KA08", 
    #            "KA09", "KA15", "KA16", "KA22", "KA30"]
    OR_folder = ["KA01", "KA03", "KA04"]
    #IR_folder = ["KI01", "KI03", "KI05", "KI07", "KI08", "KI16", "KI17", 
    #            "KI18", "KI21"]
    IR_folder = ["KI01", "KI03", "KI05"]
    MIX_folder = ["KB23", "KB24", "KB27", "KI14"] # VERIFICAR

    settings_files = ["N15_M07_F10_", "N09_M07_F10_", "N15_M01_F10_", "N15_M07_F04_"]

    n = 20

    # Normal
    for folder in normal_folder:
      for idx, setting in enumerate(settings_files):
        for i in range(1, n+1):
          key = "Normal_" + folder + "_" + str(idx) + "_" + str(i)
          files_path[key] = os.path.join(self.rawfilesdir, folder, setting + folder +
                                        "_" + str(i) + ".mat")

    # OR
    for folder in OR_folder:
      for idx, setting in enumerate(settings_files):
        for i in range(1, n+1):
          key = "OR_" + folder + "_" + str(idx) + "_" + str(i)
          files_path[key] = os.path.join(self.rawfilesdir, folder, setting + folder +
                                        "_" + str(i) + ".mat")

    # IR
    for folder in IR_folder:
      for idx, setting in enumerate(settings_files):
        for i in range(1, n+1):
          key = "IR_" + folder + "_" + str(idx) + "_" + str(i)
          files_path[key] = os.path.join(self.rawfilesdir, folder, setting + folder +
                                        "_" + str(i) + ".mat")

    self.files = files_path

  def download(self):
    """
    Download and extract compressed files from Paderborn website.
    """
    
    # RAR Files names
    if self.debug==0:
      rar_files_name = ["K001.rar","K002.rar","K003.rar","K004.rar","K005.rar","K006.rar",
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
        urllib.request.urlretrieve(url+file_name, os.path.join(dirname, dir_rar, file_name))
      print(file_name)
    
    print("Extracting files:")
    for i in rar_files_name:
      if not os.path.exists(os.path.join(dirname, i[:4])):
        file_name = os.path.join(dirname, dir_rar, i)
        Archive(file_name).extractall(dirname)  
        print(i)

    if self.debug==0:
      files_path = self.files
    else:
      files_path = files_debug(self.rawfilesdir)

    print(files_path)
    self.files = files_path

  def acquisitions(self):
    """
    Extracts the acquisitions of each file in the dictionary files_names.
    The user must be careful because this function converts all files
    in the files_names in numpy arrays.
    As large the number of entries in files_names 
    as large will be the space of memory necessary.
    
    Returns
    -------
    acquisitions : dict
    Returns the sample rate, the sample size, the conditions dict, the destinations directory
    and the acquisitions dict, where the keys represent the bearing code, followed by the conditions, by an
    algarism representing the setting and end with an algarism representing 
    the sample sequential. All features are separated by an underscore character.
    """

    acquisitions_dict = {}
    for key in self.files:
      if key != 'OR_KA08_2_2':    
        print(self.files[key])
        matlab_file = scipy.io.loadmat(self.files[key])
        if len(self.files[key])>41:
          vibration_data=matlab_file[self.files[key][19:38]]['Y'][0][0][0][6][2]
        else:
          vibration_data=matlab_file[self.files[key][19:37]]['Y'][0][0][0][6][2]

      acquisitions_dict[key] = vibration_data[0]

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

    pickle_file = 'paderborn.pickle'


    if os.path.isfile(pickle_file):
      with open(pickle_file, 'rb') as handle:
        acquisitions = pickle.load(handle)
    else:
      self.download()
      acquisitions = self.acquisitions()
      with open(pickle_file, 'wb') as handle:
        pickle.dump(acquisitions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return acquisitions
