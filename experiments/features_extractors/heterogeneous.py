
import numpy as np
from sklearn.base import TransformerMixin
from features_extractors.statisticaltime import StatisticalTime
from features_extractors.statisticalfrequency import StatisticalFrequency
from features_extractors.wavelet import WaveletPackage

class Heterogeneous(TransformerMixin):
  '''
  Extracts statistical features from both time and frequency domain.
  '''
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    st = StatisticalTime()
    stfeats = st.transform(X)
    sf = StatisticalFrequency()
    sffeats = sf.transform(X)
    wp = WaveletPackage()
    wpfeats = wp.transform(X)
    return np.concatenate((stfeats, sffeats, wpfeats), axis=1)
