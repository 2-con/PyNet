"""
Classifier
=====
  Classification algorithms for PyNet, even though some of these algorithms are can be used for regression,
  any label is considered a class and will be treated as such.
"""

"""
To anyone reading this,

I had a severe case of burnout and only finsihed some of the classifiers.
I will be working on the rest of the classifiers in the future, but for now, I will leave this here.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.arraytools import shape, transpose
from core.loss import Entropy, Gini_impurity
from core.structure import Node, Datacontainer

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
    Args
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

    distances = []
    
    for feature in self.features:
      if len(feature) != len(point): # ensure feature and point are the same length
        raise SystemError(f"Feature and point must be the same length, {feature} ({len(feature)}) is incompatible with {point} ({len(point)})")
      
      elif len(shape(feature)) != 2 or len(shape(point)) != 2: # ensures feature and point are vectors (multidimentional points)
        raise SystemError(f"Feature and point must be a vector (list of intigers or floats)")
      
      else: # calculate the manhattan distance between the point and the feature
        
        distances.append( sum(abs(a - b) for a, b in zip(feature, point)) )
        
    # sort the distances and get the indices of the k-nearest neighbors
    indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.neighbors]
    
    # get the labels of the k-nearest neighbors
    neighbors_labels = [self.labels[i] for i in indices]
    
    # return the most common label
    return max(set(neighbors_labels), key=neighbors_labels.count)

class DecisionTree:
  def __init__(self, depth:int):
    """
    Random Forest
    -----
      Predicts the class of a point based on a decision tree
    -----
    Args
    -----
    - depth (int) : the maximum depth of the decision tree
    """
    self.depth = depth
  
  def compile(self, loss:str, split:int, **kwargs):
    """
    Compile
    -----
      Compiles the model to be ready for building, setting the loss function (criterion) and split criteria.
      'infomration gain' is an alias for 'entropy'. it will be converted to 'entropy' automatically.
    -----
    Args
    -----
    - loss (str)  : the loss function to use
    - split (int) : The minimum number of samples required to split an internal node. If a node has fewer samples than this, it becomes a leaf node.
    - (Optional) random_state (int) : seed for the random number generator for tie-breaking in the decision tree, defaults to None.
    - (Optional) max_features (int) : the maximum number of features to consider when looking for the best split, defaults to None (all features).
    
    Losses
    - Gini
    - Entropy
    """
    
    self.is_compiled = True
    self.loss = loss.lower()
    self.split = split
    self.ranom_state = kwargs.get('random_state', None)
    self.max_features = kwargs.get('max_features', None)
    self.tree = None
    
    if self.loss not in ['gini', 'entropy']:
      if self.loss == 'information gain':
        self.loss = 'entropy'
      else:
        raise SystemError(f"Loss function must be 'Gini' or 'Entropy' and not {self.loss}")
  
  def fit(self, features:list, labels:list):
    """
    Fit
    -----
      Fits the model to the data, building the decision tree.
    -----
    Args:
    -----
    - features (list) : the features of the dataset, must be an array of coordinates (2D array)
    - labels   (list) : the labels of the dataset, must be an array of classes, it can be of any type
    """
    
    class container(Datacontainer):
      """
      data = coordinates
      c = class
      """
      def __init__(self, data, c, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.c = c
    
    # check the data
    if len(shape(features)) != 2:
      raise TypeError(f"Features must be a 2D array, got a {shape(features)}")
    if len(shape(labels)) != 1:
      raise TypeError(f"Features must be a 1D array, got {shape(features)}")
    if not self.is_compiled:
      raise SystemError("Model must be compiled before fitting")
    
    # setting up variables
    classes = set(labels)
    transposed_features = transpose(features)
    purity = Gini_impurity if self.loss == 'gini' else Entropy
    data = [container(f, l) for f,l in zip(features, labels)] # Datacontainer obj, dont forget!
    
    best_feature = None
    best_threshold = None
    self.tree = Node(data=labels, decision=None)
    
    leaves = [] # list of leaves to be sliced
    passed_data = [] # how much of the data is greater than the threshold
    
    """
    - using root, find leaves
    - select index 0 of 'leaves':
    
      - find the impurity
      - if impurity is 0
        - pop index 0 of 'leaves'
        - BREAK
      
      - if leaf depth is capped
        - leaf classifies as the majority class in that leaf, if tie, random
        - pop index 0 of 'leaves'
        - BREAK
      
      - for every row in feature.T:
        - from left to right, select a threshhold
        - sort the parent node's data by the threshold
        - calculate impurity
        - (min value algoritm)
        
      - use the best threshold to split the node, make sure there is enough points to split tho
    
      - create two child nodes, one for each side of the split
      - pop index 0 of 'leaves'
    
    -----
                      navigation
          vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    Node( index of the column, threshold , data = points in here, class = class of the node (defaults to None) )
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                                           training

    """
    
    # build the tree
    
    while True:
    
  def predict(self, point:list):
    """
    Predict
    -----
      Predicts the class of a point, the model must be compiled before using this method.
    -----
    Args:
    -----
    - point (list) : the point to predict, a vector (list of intigers or floats)
    """
    ...

class RandomForest:
  ...

class NaiveBayes:
  ...

class SVM:
  def __init__(self, kernel):
    """
    Support Vector Machine
    -----
      Predicts the class of a point based on a support vector machine.
    -----
    Args
    -----
    - kernel (str) : the kernel to use
    
    Kernels
    - linear
    """
    self.kernel = kernel
  
  def compile(self, C):
    """
    Compile
    -----
      Compiles the model to be ready for training.
    -----
    Args:
    -----
    - C (float) : the regularization parameter
    """
    self.C = C
    
  def fit(self, features:list, labels:list):
    """
    Fit
    -----
      Fits the model to the data, training the support vector machine.
    -----
    Args:
    -----
    - features (list) : the features of the dataset, must be a 2D array
    - labels   (list) : the labels of the dataset, must be a 1D array
    """
    
    # check the data
    if len(shape(features)) != 2:
      raise TypeError(f"Features must be a 2D array, got {shape(features)}")
    if len(shape(labels)) != 1:
      raise TypeError(f"Labels must be a 1D array, got {shape(labels)}")
    
    # training the SVM
    ...
    
    def predict(self, point:list):
      """
      Predict
      -----
        Predicts the class of a point, the model must be compiled before using this method.
      -----
      Args:
      -----
      - point (list) : the point to predict, a vector (list of intigers or floats)
      """
      ...
      