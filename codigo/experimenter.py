from sklearn.model_selection import cross_validate, GroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from database import Database_Experimenter

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

"""
Class definition of dataset segmentation and experiments.
"""
debug = 0

class Experimenter(Database_Experimenter):
  """
  Datasets class wrapper for experiment framework.

  ...
  Attributes
  ----------
  acquisitions : dict
    Dictionary with the sample_rate, sample_size, conditions, dirdest and acquisitions
  
  Methods
  -------
  segmentate()
    Semgmentate the raw files.
  perform()
    Perform experiments.

  """
  def __init__(self, acquisitions, sample_size):
    self.sample_size = sample_size
    self.acquisitions = acquisitions['acquisitions']

  def segmentate(self):
    """
    Segmentate files by the conditions and returns signals data, signals
    condition and signals acquisition.
    """
    
    n = len(self.acquisitions)

    self.signal_dt = np.empty((0,self.sample_size))
    self.signal_or = []
    self.signal_gr = []

    for i,key in enumerate(self.acquisitions):
      acquisition_size = len(self.acquisitions[key])
      if debug:
        n_samples = 15
      else:
        n_samples = acquisition_size//self.sample_size
      print('{}/{} --- {}: {}'.format(i+1, n, key, n_samples))
      self.signal_dt = np.concatenate((self.signal_dt, self.acquisitions[key][:(n_samples*self.sample_size)].reshape(
          (n_samples,self.sample_size))))
      for j in range(n_samples):
        self.signal_gr.append(key)
        self.signal_or.append(key[0])


  def perform(self, clfs, scoring, verbose=0):
    
    self.segmentate()

    # Estimators
    for clf_name, estimator in clfs:
      print("*"*(len(clf_name)+8),'\n***',clf_name,'***\n'+"*"*(len(clf_name)+8))

      print("---Kfold---")
      estimator_kfold = estimator

      score = cross_validate(estimator_kfold, self.signal_dt, np.array(self.signal_or),
                            scoring=scoring, cv=4, verbose=verbose)#, error_score='raise')
      for metric,s in score.items():
        print(metric, ' \t', s, ' Mean: ', format(s.mean(), '.2f'), ' Std: ', format(s.std(), '.2f'))

      print("---GroupKfold---")
      estimator_groupkfold = estimator
      score = cross_validate(estimator_groupkfold, self.signal_dt, np.array(self.signal_or),
                             self.signal_gr, scoring, cv=GroupKFold(n_splits=4), verbose=verbose)

      for metric, s in score.items():
        print(metric, ' \t', s, ' Mean: ', format(s.mean(), '.2f'), ' Std: ', format(s.std(), '.2f'))
