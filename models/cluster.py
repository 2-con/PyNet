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
  def fit(self, clusters, points, **kwargs):
    epochs = kwargs.get('epochs', 10)

    # find the k means
    for _ in range(epochs):
      pass

  def predict(self, point):
    pass

class DBScan:
  def fit(self, points, **kwargs):
    epochs = kwargs.get('epochs', 10)

  def predict(self, point):
    pass
