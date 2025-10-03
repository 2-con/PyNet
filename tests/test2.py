import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.classifier import *
# from tools.visual import display_boundary

training_features = [
  [5,0],
  [2,2],
  [4,10],
  [3,8],
]

training_target = [0,0,1,1]

model = NaiveBayes()
model.compile("multinomial")
model.fit(training_features, training_target)
model.predict([4,10])

# display_boundary(model, training_features, training_target)