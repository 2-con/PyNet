import sys
import os

# Get the directory of the current script (test.py)
current_script_dir = os.path.dirname(__file__)

# Navigate up one level to the 'PyNet' directory
# If test.py is in PyNet/tests/, then '..' takes us to PyNet/
pynet_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))

# Add the PyNet root directory to Python's module search path
sys.path.append(pynet_root_dir)
import api.synapse as ai

import tools.arraytools as tools
import tools.visual as visual
from tools.arraytools import generate_random_array

test = 3

if test == 1: # CNN

  up = [
    [1,1,1],
    [0,0,0],
    [0,0,0]
  ]
  down = [
    [0,0,0],
    [0,0,0],
    [1,1,1]
  ]
  left = [
    [1,0,0],
    [1,0,0],
    [1,0,0]
  ]
  right = [
    [0,0,1],
    [0,0,1],
    [0,0,1]
  ]
  
  trainX = [up, down, left, right]
  trainY = [
    [1,0,0,0], 
    [0,1,0,0], 
    [0,0,1,0], 
    [0,0,0,1]
    ]
  
  model = ai.Sequential( 
    
    ai.Convolution( kernel=generate_random_array(2,2), activation='none', bias=True, learnable=True ),
    ai.Flatten(),
  )

  model.compile(
    optimizer='default', 
    loss='mean squared error', 
    learning_rate=0.1, 
    epochs=10000,
    metrics=['accuracy']
  )
  
  model.fit(
    trainX, 
    trainY, 
    regularity=1, 
    verbose=4
  )

  print(model.push(up))
  # print([1,0])

elif test == 2: # NN
  training_features = [[0,0],[0,1],[1,0],[1,1]]
  training_target   = [[0,1],[1,0],[1,0],[0,1]]

  model = ai.Sequential(
    ai.Dense(4, activation='elu'),
    ai.Dense(3, activation='elu'),
    ai.Dense(2, activation='elu'),
  )

  model.compile(
    optimizer='default', 
    loss='mean squared error', 
    learning_rate=0.1, 
    batch_size=1,
    epochs=20000, 
    metrics=['accuracy']
  ) 

  model.fit(training_features, training_target, regularity=1000, verbose=3)
  
  for feature, target in zip(training_features, training_target):
    print(f"pred {model.push(feature)} true {target}")
  
elif test == 3: # RNN
  training_features = [
    [2,3,4,5], 
    [0,1,2,3], 
    [4,5,6,7], 
    [-2,-1,0,1]
    ]
  training_target   = [
    [6],
    [4],
    [8],
    [2]]

  model = ai.Sequential(
    ai.Recurrent('none', input=True, output=False),
    ai.Recurrent('none', input=True, output=False),
    ai.Recurrent('none', input=True, output=False),
    ai.Recurrent('none', input=True, output=True),
  )
  
  model.compile(
    optimizer='default', 
    loss='mean squared error', 
    learning_rate=0.02,
    batch_size=1, 
    epochs=10000,
    metrics=['accuracy']
  ) 
  
  model.fit(training_features, training_target, regularity=1, verbose=4)
  
  for feature, target in zip(training_features, training_target):
    print(f"pred {model.push(feature)} true {target}")
    
  print(f"{model.push([8,9,10,11])} true [12]")

elif test == 4: # LSTM
  training_features = [
    [2,3,4,5], 
    [0,1,2,3], 
    [4,5,6,7], 
    [-2,-1,0,1]
    ]
  training_target   = [
    [6],
    [4],
    [8],
    [2]]

  model = ai.Sequential(
    ai.LSTM(),
    ai.LSTM(),
    ai.LSTM(),
    ai.LSTM(),
  )
  
  model.compile(
    optimizer='default', 
    loss='mean squared error', 
    learning_rate=0.02,
    batch_size=1, 
    epochs=10000,
    metrics=['accuracy']
  ) 
  
  model.fit(training_features, training_target, regularity=1, verbose=4)
  
  for feature, target in zip(training_features, training_target):
    print(f"pred {model.push(feature)} true {target}")
    
  print(f"{model.push([8,9,10,11])} true [12]")