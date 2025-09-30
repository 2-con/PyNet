import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.classifier import SVM
from tools.visual import display_boundary
from sklearn.datasets import make_moons
import core.vanilla.kernel as Kernel

training_features, training_target = make_moons(n_samples=200, noise=0.12, random_state=2)

training_features = training_features.tolist()
training_target = [x for x in training_target.tolist()]

model = SVM()
model.compile(Kernel.Polynomial(1, 3), 100, 1)
model.fit(training_features, training_target)

display_boundary(model, training_features, training_target)