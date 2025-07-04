"""
Classifier
=====

"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.arraytools import shape

class KNN:
  def __init__(self, features, labels):
    """
    K-Nearest Neighbors
    -----
      Predicts the class of a point based on its k-nearest neighbors
    -----
    Args
    -----
    - features (list) : the features of the dataset, must be a 2D array
    - labels   (list) : the labels of the dataset, must be a 2D array
    """
    self.features = features
    self.labels = labels
    self.is_compiled = False
    
  def compile(self, neighbors):
    """
    Compile
    -----
      Compiles the model to be ready for training, setting the number of neighbors to consider
    -----
    Args:
    -----
    - neighbors (int) : the number of neighbors to consider before classifying
    """
    self.neighbors = neighbors
    self.is_compiled = True

  def predict(self, point):
    """
    Predict
    -----
      Predicts the class of a point, the model must be compiled before using this method.
    -----
    Args:
    -----
    - point (list) : the point to predict, a vector (list of intigers or floats)
    """
    
    if not self.is_compiled:
      raise SystemError("Model must be compiled before predicting anything")

    for feature in self.features:
      if len(feature) != len(point):
        raise SystemError(f"Feature and point must be the same length, {feature} ({len(feature)}) is incompatible with {point} ({len(point)})")
      elif len(shape(feature)) != 2 or len(shape(point)) != 2:
        raise SystemError(f"Feature and point must be a vector (list of intigers or floats)")
      else:
        pass
      
    pass

class SVM:
  ...

class NaiveBayes:
  ...

class RandomForest:
  ...

class DecisionTree:
  ...
