"""
Cluster
=====
  Unsupervised learning algorithms for clustering data points.
-----
Provides
-----
- KMeans
- DBScan
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

  def cluster(self, point):
    pass

class Kmedoid:
  def __init__(self):
    """
    K-Medoid clustering algorithm
    """
    
    self.centroids = []
    self.is_fitted = False
  
  def compile(self, *args, **kwargs):
    self.is_compiled = True
    
  def fit(self, points, **kwargs):
    # find the k medoids
    ...
  def cluster(self, point):
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
  def cluster(self, point):
    pass
