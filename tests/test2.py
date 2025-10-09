import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.cluster import *
from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
from tools.arraytools import transpose

# training_features = make_blobs(n_samples=10, centers=2, n_features=2, random_state=0, cluster_std=0.01)[0].tolist()
training_features = make_moons(n_samples=500, noise=0.1)[0].tolist()

model = DBScan()
model.compile(0.15, 2, 2)
model.fit(training_features)
plt.scatter(transpose(training_features)[0], transpose(training_features)[1], c=model.classify())
plt.grid(True)
plt.show()

