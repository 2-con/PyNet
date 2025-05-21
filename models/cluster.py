"""
Cluster
=====

"""
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
