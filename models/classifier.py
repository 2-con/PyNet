"""
Classifier
=====
  Classification algorithms for PyNet, even though some of these algorithms are can be used for regression,
  any label is considered a class and will be treated as such.
-----
Provides
-----
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Naive Bayes
- SVM
"""

"""
might add adaboost at some point
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.arraytools import shape, transpose
from core.loss import Entropy, Gini_impurity
from core.datafield import Node as n
from core.datafield import Datacontainer as dc
import random
import math

class KNN:
  def __init__(self, ):
    """
    K-Nearest Neighbors
    -----
      Predicts the class of a point based on its k-nearest neighbors
    """
    self.is_compiled = False
    
  def compile(self, features, labels, neighbors):
    """
    Compile
    -----
      Compiles the model to be ready for training, setting the number of neighbors to consider
    -----
    Args
    -----
    - features (list) : the features of the dataset, must be a 2D array
    - labels   (list) : the labels of the dataset, must be a 2D array
    - neighbors (int) : the number of neighbors to consider before classifying
    """
    self.is_compiled = True
    self.neighbors = neighbors
    self.features = features
    self.labels = labels
    
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
      
      elif len(shape(point)) != 1: # ensures feature and point are vectors (multidimentional points)
        raise SystemError(f"point must be a vector (list of intigers or floats)")
      
      else: # calculate the manhattan distance between the point and the feature
        
        distances.append( sum(abs(a - b) for a, b in zip(feature, point)) )
        
    # sort the distances and get the indices of the k-nearest neighbors
    indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.neighbors]
    
    # get the labels of the k-nearest neighbors
    neighbors_labels = [self.labels[i] for i in indices]
    
    # return the most common label
    return max(set(neighbors_labels), key=neighbors_labels.count)

class DecisionTree:
  def __init__(self):
    """
    Random Forest
    -----
      Predicts the class of a point based on a decision tree
    """
    self.is_compiled = False
    self.is_trained = False
  
  def compile(self, depth:int, loss:str, split:int, **kwargs):
    """
    Compile
    -----
      Compiles the model to be ready for building.
      'infomration gain' is an alias for 'entropy'. it will be converted to 'entropy' automatically.
    -----
    Args
    -----
    - loss (str)  : the loss function to use
    - split (int) : The minimum number of samples required to split an internal node. If a node has fewer samples than this, it becomes a leaf node.
    - (Optional) max_features (int) : The maximum number of features to consider when looking for the best split, defaults to 'All
    
    Losses
    - Gini
    - Entropy
    
    Max Features
    - All
    - Sqrt
    - Log2
    """
    
    self.depth = depth
    self.loss = loss.lower()
    self.split = split
    self.max_features:str = kwargs.get('max_features', 'all').lower()
    
    self.tree = None
    self.is_compiled = True
    self.is_trained = False
    
    if self.loss not in ['gini', 'entropy']:
      if self.loss == 'information gain':
        self.loss = 'entropy'
      else:
        raise SystemError(f"Loss function must be 'Gini' or 'Entropy' and not {self.loss}")
    
    if self.max_features.lower() not in ('all', 'sqrt', 'log2'):
      raise SystemError(f"Max features must be 'All', 'Sqrt' or 'Log2' and not {self.max_features}")
  
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
    
    self.is_trained = True
    
    class container(dc):
      """
      data = coordinates
      c = class
      """
      def __init__(self, data, c, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.c = c
      
    class Node(n):
      def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
    
    # check the data
    if len(shape(features)) != 2:
      raise TypeError(f"Features must be a 2D array, got a {shape(features)}")
    if len(shape(labels)) != 1:
      raise TypeError(f"Labels must be a 1D array, got {shape(features)}")
    if not self.is_compiled:
      raise SystemError("Model must be compiled before fitting")
    
    # setting up variables
    transposed_features = transpose(features)
    purity = Gini_impurity if self.loss == 'gini' else Entropy
    data = [container(f, l) for f,l in zip(features, labels)] # Datacontainer obj, dont forget!
    
    best_feature = 0 # the best feature to split by so far
    best_threshold = 0 # the best threshold so far
    self.tree = Node(data, category=None, threshold=None, decision=None)
    
    leaves = [] # list of leaves to be sliced
    passed_data_left = [] # how much of the data is greater than the threshold
    passed_data_right = [] # how much of the data is lesser than the threshold
    
    stop = False
    
    if self.max_features == 'all':
      max_features = len(transposed_features)
    elif self.max_features == 'sqrt':
      max_features = math.ceil(math.sqrt(len(transposed_features)))
    elif self.max_features == 'log2':
      max_features = math.ceil(math.log2(len(transposed_features)))

    # build the tree uisng BFS
    while not stop:
      
      leaves = self.tree.get_last()
      for leaf in leaves:
        
        impurity = purity([x.c for x in leaf.data])
        
        # iterate over all the features
        
        max_features_valid = [random.randint(0, len(transposed_features) - 1) for _ in range(max_features)]
        
        for feature_index, feature in enumerate(transposed_features):
          
          if feature_index not in max_features_valid and self.max_features != 'all':
            continue
          
          # iterate over all the thresholds possible
          for threshold in feature:
            
            # get the classes of the points that pass and fail the threshold
            passed_data_left = [x.c for x in leaf.data if x.data[feature_index] > threshold]
            passed_data_right = [x.c for x in leaf.data if x.data[feature_index] < threshold]
            
            # if theres enough data to consider a split
            if len(passed_data_left) > 0:
              
              new_impurity_left = purity(passed_data_left)
              new_impurity_right = purity(passed_data_right)
              new_impurity = ((len(passed_data_left) / len(leaf.data)) * new_impurity_left + (len(passed_data_right) / len(leaf.data)) * new_impurity_right)
              
              # min value algorithm
              if new_impurity < impurity:
                impurity = new_impurity
                best_feature = feature_index
                best_threshold = threshold
        
        leaf.kwargs['category'] = best_feature      
        leaf.kwargs['threshold'] = best_threshold
        
        # split the node if there is enough data
        if len(leaf.data) >= self.split:
          
          left = Node([x for x in leaf.data if x.data[best_feature] > best_threshold], category=None, threshold=None, decision=None)
          right = Node([x for x in leaf.data if x.data[best_feature] <= best_threshold], category=None, threshold=None, decision=None)
          
          leaf.add_child(left)
          leaf.add_child(right)
        
        if leaf.depth() == self.depth or len(leaf.data) < self.split:
          stop=True
  
    # set classification
    
    for leaf in self.tree.get_last():
      if leaf.kwargs['decision'] is None:
        leaf.kwargs['decision'] = max(set([x.c for x in leaf.data]), key=[x.c for x in leaf.data].count) if len(leaf.data) > 0 else 0
    
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
    
    if not self.is_compiled:
      raise SystemError("Model must be compiled before predicting anything")
    if not self.is_trained:
      raise SystemError("Model must be trained before predicting anything")
    
    current_node = self.tree
    
    while True:
      
      # check if the node is a leaf
      if current_node.kwargs['decision'] is not None:
        return current_node.kwargs['decision']
      
      else:
        # check if the category of the point is greater than the threshold
        if point[current_node.kwargs['category']] > current_node.kwargs['threshold']:
          
          # left = bigger & right = smaller
          current_node = current_node.children[0]
        else:
          current_node = current_node.children[1]

class RandomForest:
  def __init__(self):
    """
    Random Forest
    -----
      Predicts the class of a point based on a decision tree
    """
    self.is_compiled = False
    self.is_trained = False
  
  def compile(self, n_estimators:int, depth:int, loss:str, split:int, **kwargs):
    """
    Compile
    -----
      Compiles the model to be ready for building.
      'infomration gain' is an alias for 'entropy'. it will be converted to 'entropy' automatically.
    -----
    Args
    -----
    - n_estimators (int) : the number of trees to build
    - loss (str)  : the loss function to use
    - split (int) : The minimum number of samples required to split an internal node. If a node has fewer samples than this, it becomes a leaf node.
    - depth (int) : The maximum depth of any of the trees
    - (Optional) max_features (str) : The maximum number of features to consider when looking for the best split
    - (Optional) bootstrap (bool) : If True, the bootstrap sample is used when building trees. If False, the whole dataset is used to build each tree. True by default
    
    Losses
    - Gini
    - Entropy 
    
    Max Features
    - All
    - Sqrt
    - Log2
    """
    self.n_estimators = n_estimators
    self.depth = depth
    self.loss = loss.lower()
    self.split = split
    
    self.max_features:str = kwargs.get('max_features', 'all').lower()
    self.bootstrap_data = kwargs.get('bootstrap', True)
    
    self.is_compiled = True
    self.is_trained = False
    self.trees = []
    
    if self.loss not in ['gini', 'entropy']:
      if self.loss == 'information gain':
        self.loss = 'entropy'
      else:
        raise SystemError(f"Loss function must be 'Gini' or 'Entropy' and not '{self.loss}'")
    
    if self.max_features.lower() not in ('all', 'sqrt', 'log2'):
      raise SystemError(f"Max features must be 'All', 'Sqrt' or 'Log2' and not '{self.max_features}'")
  
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
    
    self.is_trained = True
    
    class container(dc):
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
      raise TypeError(f"Labels must be a 1D array, got {shape(features)}")
    if not self.is_compiled:
      raise SystemError("Model must be compiled before fitting")

    # tree maker
    def build_tree(features:list, labels:list, depth:int, loss:str, split:int) -> n:
      """
      Internal class, returns a decision tree
      """
      
      tree = DecisionTree()
      tree.compile(
        depth=depth,
        loss=loss,
        split=split,
        max_features=self.max_features
      )
      tree.fit(features, labels)
      return tree
    
    data = [container(f, l) for f,l in zip(features, labels)] # Datacontainer obj, dont forget!
    data_per_tree = []
    
    # bootstrap data
    if self.bootstrap_data:
      for _ in range(self.n_estimators):
        bootstraped_data = []
        
        for _ in range(len(data)):
          bootstraped_data.append(random.choice(data))
        
        data_per_tree.append(bootstraped_data)
    
    else:
      data_per_tree = [data for _ in range(self.n_estimators)]
    
    for data in data_per_tree:
      tree = build_tree(
        features=[d.data for d in data],
        labels=[d.c for d in data],
        depth=self.depth,
        loss=self.loss,
        split=self.split
      )
      self.trees.append(tree)
    
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
    
    if not self.is_compiled:
      raise SystemError("Model must be compiled before predicting anything")
    if not self.is_trained:
      raise SystemError("Model must be trained before predicting anything")
    
    votes = []
    vote_count = {}
    
    for tree in self.trees:
      votes.append(tree.predict(point))
    
    # tally up the votes
    
    for vote in votes:
      if vote in vote_count:
        vote_count[vote] += 1
      else:
        vote_count[vote] = 1
    
    return max(vote_count, key=vote_count.get)

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
