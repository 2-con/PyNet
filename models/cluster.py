"""
Cluster
=====

"""



"""
To anyone reading this,

This wasn't part of the original plan, but I decided to add clustering algorithms to the models.
I will be working on the rest of the clustering algorithms in the future, but for now, I will leave this here.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class KMeans:
  def __init__(self):
    """
    K-Means clustering algorithm
    """
    
    self.centroids = []
    self.is_fitted = False
  
  def compile(self, *args, **kwargs):
    self.is_compiled = True
    
  def fit(self, points, **kwargs):

    # find the k means
    ...

  def predict(self, point):
    pass

class DBScan:
  def __init__(self):
    """
    DBScan clustering algorithm
    """
    
    self.centroids = []
    self.is_fitted = False
  
  def compile(self, *args, **kwargs):
    self.is_compiled = True
    
  def fit(self, points, **kwargs):
    # find the clusters
    ...
  def predict(self, point):
    pass
