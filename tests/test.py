import sys
import os

# Get the directory of the current script (test.py)
current_script_dir = os.path.dirname(__file__)
pynet_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(pynet_root_dir)

import api.synapse as ai

import tools.arraytools as tools
import tools.visual as visual
from tools.arraytools import generate_random_array
import matplotlib.pyplot as plt

test = 6

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
    [1,1,0,0], 
    [0,0,1,1], 
    [1,0,1,0], 
    [0,1,0,1]
    ]
  
  model = ai.Sequential( 
    
    ai.Convolution((2,2), 'relu'),
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
    regularity=1000, 
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
    optimizer='none', 
    loss='mean squared error', 
    learning_rate=0.01,
    batch_size=1,
    epochs=10000, 
    metrics=['accuracy'],
  ) 

  model.fit(training_features, training_target, regularity=100, verbose=4)
  
  for feature, target in zip(training_features, training_target):
    print(f"pred {model.push(feature)} true {target}")
  
elif test == 3: # RNN
  training_features = [
    
    [1, 2, 3, 4],   # 10
    [0, 1, 2, 3],   # 6
    [5, 5, 0, 0],   # 10
    [1, 1, 1, 1],   # 4
    [2, 2, 1, 2]    # 7
  ]

  training_target = [
    [10],
    [6],
    [10],
    [4],
    [7]
  ]

  model = ai.Sequential(
    ai.Recurrent('none', input=True, output=False),
    ai.Recurrent('none', input=True, output=False),
    ai.Recurrent('none', input=True, output=False),
    ai.Recurrent('none', input=True, output=False),
    ai.Recurrent('none', input=False, output=True),
  )
  
  model.compile(
    optimizer='rmsprop', 
    loss='mean squared error', 
    learning_rate=0.00001,
    batch_size=3, 
    epochs=300000,
    metrics=['accuracy']
  ) 
  
  model.fit(training_features, training_target, regularity=100, verbose=4)
  
  for feature, target in zip(training_features, training_target):
    print(f"pred {model.push(feature)} true {target}")
    
  print(f"{model.push([5,5,5,5])} true [20]")

elif test == 4: # LSTM
  training_features = [
    
    [1, 2, 3, 4],   # 10
    [0, 1, 2, 3],   # 6
    [5, 5, 0, 0],   # 10
    [1, 1, 1, 1],   # 4
    [2, 2, 1, 2]    # 7
  ]

  training_target = [
    [1],
    [-1/3],
    [1],
    [-1],
    [0]
  ]

  model = ai.Sequential(
    ai.LSTM(input=True, output = False),
    ai.LSTM(input=True, output = False),
    ai.LSTM(input=True, output = False),
    ai.LSTM(input=True, output = False),
    ai.LSTM(input=False, output = True),
  )
  
  model.compile(
    optimizer='rmsprop', 
    loss='mean squared error', 
    learning_rate=0.0001,
    batch_size=3, 
    epochs=50000,
    metrics=['accuracy']
  ) 
  
  model.fit(
    training_features, 
    training_target, 
    regularity=100, 
    verbose=4
    )
   
  for feature, target in zip(training_features, training_target):
    print(f"pred {model.push(feature)} true {target}")

elif test == 5: # GRU
  training_features = [
    
    [1, 2, 3, 4],   # 10
    [0, 1, 2, 3],   # 6
    [5, 5, 0, 0],   # 10
    [1, 1, 1, 1],   # 4
    [2, 2, 1, 2]    # 7
  ]

  training_target = [
    [1],
    [-1/3],
    [1],
    [-1],
    [0]
  ]

  model = ai.Sequential(
    ai.GRU(input=True, output = False),
    ai.GRU(input=True, output = False),
    ai.GRU(input=True, output = False),
    ai.GRU(input=True, output = False),
    ai.GRU(input=False, output = True),
  )
  
  model.compile(
    optimizer='adam', 
    loss='mean squared error', 
    learning_rate=0.001,
    batch_size=1, 
    epochs=50000,
    metrics=['accuracy']
  )  
  
  model.fit(
    training_features, 
    training_target, 
    regularity=100, 
    verbose=4
    )
   
  for feature, target in zip(training_features, training_target):
    print(f"pred {model.push(feature)} true {target}")

elif test == 6: # Paralellization test
  
  training_features = [
    
    generate_random_array(20,20)
    
  ]

  training_target = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  ]
  
  KRNL = (2,2)
  
  model = ai.Sequential( 
    
    ai.Parallel(
    ai.Convolution(KRNL, 'elu'), ai.Convolution(KRNL, 'elu'), ai.Convolution(KRNL, 'elu'), ai.Convolution(KRNL, 'elu')
    ),
  ai.Parallel(
    ai.Convolution(KRNL, 'elu'), ai.Convolution(KRNL, 'elu'), ai.Convolution(KRNL, 'elu'), ai.Convolution(KRNL, 'elu')
    ),
  ai.Parallel(
    ai.Convolution(KRNL, 'elu'), ai.Convolution(KRNL, 'elu'), ai.Convolution(KRNL, 'elu'), ai.Convolution(KRNL, 'elu')
    ),
  ai.Parallel(
    ai.Flatten(), ai.Flatten(), ai.Flatten(), ai.Flatten(),
    ),
  ai.Merge('concat'),
  ai.Dense(16, 'elu'),
  ai.Dense(16, 'elu'),
  ai.Dense(10, 'elu'),
  ai.Operation('softmax')
  )

  model.compile(
    optimizer='adam', 
    loss='mean squared error', 
    learning_rate=0.01,
    batch_size=1,
    epochs=500,
    metrics=['accuracy']
  )
  
  model.fit(
    training_features, 
    training_target, 
    regularity=100, 
    verbose=4
  )
  
  for feature, target in zip(training_features, training_target):
    print(f"pred {model.push(feature)} true {target}")

plt.plot(range(len(model.error_logs)), model.error_logs)
plt.title("Model Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel(f" {model.loss} Loss")
plt.show()