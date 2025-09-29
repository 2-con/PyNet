import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.classifier import SVM
from tools.visual import display_boundary
from sklearn.datasets import make_blobs


training_features, training_target = make_blobs(n_samples=50, n_features=2, centers=2, random_state=0, cluster_std=0.5)

training_features = training_features.tolist()
training_target = [x for x in training_target.tolist()]
training_features.append([0,0.2])
training_target.append(0)

def linear_kernel(x1, x2):
  return sum([a*b for a,b in zip(x1, x2)])

model = SVM()
model.compile(100, 1, linear_kernel, 1)
model.fit(training_features, training_target)

print(model.predict([0,0.2]))
print(training_target[-1])

display_boundary(model, training_features, training_target)