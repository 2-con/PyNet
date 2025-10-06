import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.classifier import *
from tools.visual import display_boundary
from sklearn.datasets import make_blobs
from core.vanilla.kernel import Linear

training_features = make_blobs(n_samples=150, centers=3, n_features=2, random_state=0, cluster_std=0.9)[0].tolist()
training_target = make_blobs(n_samples=150, centers=3, n_features=2, random_state=0)[1].tolist()

model = MSVM()
model.compile(Linear(), 100, 1)
model.fit(training_features, training_target)
print(model.predict([0,0]))

display_boundary(model, training_features, training_target)