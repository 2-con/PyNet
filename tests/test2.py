import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api.synapse as net
from tools.arraytools import generate_random_array as array
from tools.visual import array_display as display

features = array(10,100)
targets = array(4,100)

model = net.Sequential(
  net.Dense(8, 'relu'),
  net.Dense(4, 'relu'),
  net.Dense(4, 'relu'),
)

model.compile(
  optimizer='adam',
  loss='mean squared error',
  learning_rate=0.01,
  epochs=100,
  metrics=['accuracy'],
  logging=10,
  verbose=3
)

model.fit(features, targets)
