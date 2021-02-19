# Experimenter

from database import Database_Experimenter
from perform.segmentator import segmentator
from perform.performer import perfomer


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

        self.signal_dt, self.signal_or, self.signal_gr = segmentator(self.acquisitions, self.sample_size, debug)

    def perform(self, clfs, scoring, verbose=0):

        self.segmentate()

        perfomer(clfs, self.signal_dt, self.signal_or, self.signal_gr, scoring, verbose)
