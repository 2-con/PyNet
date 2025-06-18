import sys
import os
current_script_dir = os.path.dirname(__file__)
pynet_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(pynet_root_dir)

from tools.arraytools import generate_random_array, flatten, reshape, shape
from tools.visual import numerical_display
import api.synapse as syn

features = generate_random_array(5,5,3)
targets = generate_random_array(4,3)

# numerical_display(features)
# numerical_display(targets)

model = syn.Sequential(
  syn.Convolution((3,3), 3, 'none'),
  syn.Convolution((2,2), 2, 'none'),
  syn.Flatten(),
  syn.Dense(4, 'relu')
)

model.compile(
  loss='mean squared error',
  epochs = 100,
  optimizer='default',
  metrics=['accuracy'],
  verbose = 4,
  learning_rate = 0.01,
  logging = 10
)

model.fit(
  features,
  targets
)