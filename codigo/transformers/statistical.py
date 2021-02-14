
import numpy as np
from sklearn.base import TransformerMixin
from transformers.statisticaltime import StatisticalTime
from transformers.statisticalfrequency import StatisticalFrequency

class Statistical(TransformerMixin):
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
    return np.concatenate((stfeats,sffeats),axis=1)
