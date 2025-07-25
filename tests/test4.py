import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.datasets import make_moons, make_blobs

# training_features, training_target = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.9)
training_features, training_target = make_moons(n_samples=200, noise=0.05)

from models.classifier import RandomForest, DecisionTree
from tools.visual import display_boundary, tree_display

features = training_features.tolist()
targets =  training_target.tolist()

# model = RandomForest()
# model.compile(
#   n_estimators=10,
#   depth=8,
#   loss='gini',
#   split=1,
#   bootstrap=True,
#   max_features='sqrt'
# )

model = DecisionTree()
model.compile(
  depth=10,
  loss='gini',
  split=1,
)

model.fit(features, targets)
display_boundary(model, features, targets, zoom=1)

