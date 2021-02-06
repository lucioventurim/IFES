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

    print(len(set(self.signal_gr)))


  def perform(self, clfs, scoring, verbose=0):
    
    self.segmentate()

    # Estimators
    for clf_name, estimator in clfs:
      print("*"*(len(clf_name)+8),'\n***',clf_name,'***\n'+"*"*(len(clf_name)+8))

      print("---Kfold---")
      estimator_kfold = estimator

      #scores = cross_val_score(estimator_kfold, self.signal_dt, np.array(self.signal_or), cv=2)

      #print("test_accuracy: ", scores, " Mean:", format(scores.mean(), '.2f'), "Std:", format(scores.std(), '.2f'))


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

      print("---Train Test Split---")
      estimator_ttsplit = estimator
      X_train, X_test, y_train, y_test = train_test_split(self.signal_dt, self.signal_or, test_size = 0.3,
                                                          random_state = 42)

      estimator_ttsplit.fit(X_train, y_train)
      pred = estimator_ttsplit.predict(X_test)

      print("test_accuracy: ", format(accuracy_score(y_test, pred), '.2f'))
      print("test_f1_macro: ", format(f1_score(y_test, pred, average='macro'), '.2f'))

      print("---Train Test Split - Group---")
      estimator_ttsplitgroup = estimator
      train_inds, test_inds = next(GroupShuffleSplit(test_size=.30,
                                                     n_splits=2,
                                                     random_state=42).split(self.signal_dt,
                                                                            groups=self.signal_gr))

      X_train = self.signal_dt[train_inds]
      X_test = self.signal_dt[test_inds]
      y_train = np.array(self.signal_or)[train_inds]
      y_test = np.array(self.signal_or)[test_inds]


      estimator_ttsplitgroup.fit(X_train, y_train)
      pred = estimator_ttsplitgroup.predict(X_test)

      print("test_accuracy: ", format(accuracy_score(y_test, pred), '.2f'))
      print("test_f1_macro: ", format(f1_score(y_test, pred, average='macro'), '.2f'))
