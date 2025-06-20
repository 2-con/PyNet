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

model = syn.Sequential( # 5x5 -> 3x3 (3) -> 2x2 (2)
  syn.Convolution((3,3), 3, 'none'),
  syn.Convolution((2,2), 2, 'none'),
  syn.Flatten(),
  syn.Dense(4, 'elu')
)

model.compile(
  loss='mean squared error',
  epochs = 1000,
  optimizer='adam',  
  metrics=['accuracy'],
  verbose = 4,
  learning_rate = 0.02,
  logging = 100
)

model.fit(
  features,
  targets
)