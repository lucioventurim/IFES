# Author: Lucio Venturim <lucioventurim@gmail.com> 
# Francisco Boldt <fboldt@gmail.com>

from abc import ABC, abstractmethod

class Database_Download(ABC): 
    """
    Base inteface to implement database download and acquisition.
    """
    
    @abstractmethod
    def download(self):
        """
        Method responsible for downloading the raw database files from source website.
        """
        pass

    @abstractmethod
    def acquisitions(self):
        """
        Method responsible for extracting the acquisitions from the database files.
        """
        pass

    @abstractmethod
    def load(self):
        """
        Method responsible for loading the data set.
        """
        pass

class Database_Experimenter(ABC): 
    """
    Base inteface to implement database wrapper classes.
    """

    @abstractmethod
    def segmentate(self):
        """
        Method responsible for segmentating the raw database files.
        """
        pass

    @abstractmethod
    def perform(self):
        """
        Method responsible for returning experiments results.
        """
        pass
