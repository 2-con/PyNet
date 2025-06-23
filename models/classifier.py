"""
Classifier
=====

"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class KNN:
  def fit(self, points: list, labels: list):
    """
    fits the model and stores the data
    """
    self.points = points
    self.labels = labels

  def predict(self, point, neighbors):
    """
    Finds the k-neighbors of the point and returns the most common label
    -----
    point (list or tuple): the point to predict
    neighbors (int): the max number of neighbors to consider
    """
    pass

class SVM:
  def fit(self, points, labels):
    """
    fits the model and stores the data
    """
    self.points = points
    self.labels = labels

  def predict(self, point):
    """
    decide where the point lie between the decicion boundary
    """
    pass

class NaiveBayes:
  def fit(self, points, labels):
    """
    fits the model and stores the data
    """
    self.points = points
    self.labels = labels

  def predict(self, point):
    """
    predicts
    """

class RandomForest:
  def fit(self, points, labels, trees, tree_depth):
    """
    fits the model and stores the data
    """
    self.points = points
    self.labels = labels

  def predict(self, point):
    """
    predicts
    """
    pass

class DecisionTree:
  def fit(self, points, labels, tree_depth):
    """
    fits the model and stores the data
    """
    self.points = points
    self.labels = labels

  def predict(self, point):
    """
    predicts
    """
    pass
