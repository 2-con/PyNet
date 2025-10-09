import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.classifier import *
from sklearn.datasets import make_blobs

training_features = make_blobs(n_samples=150, centers=3, n_features=2, random_state=0, cluster_std=0.9)[0].tolist()

