import sys
import os

# Get the directory of the current script (test.py)
current_script_dir = os.path.dirname(__file__)
pynet_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(pynet_root_dir)

import api.synapse as net

import tools.arraytools as tools
import tools.visual as visual
from tools.arraytools import generate_random_array
import matplotlib.pyplot as plt
import time
import math
import random

start_time = time.perf_counter()
test = 7

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
  
  model = net.Sequential( 
    
    net.Convolution((2,2), 'relu'),
    net.Flatten(),
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
  )

  print(model.push(up))
  # print([1,0])

elif test == 2: # NN
  training_features = [[0,0],[0,1],[1,0],[1,1]]
  training_target   = [[0,1],[1,0],[1,0],[0,1]]

  model = net.Sequential(
    net.Dense(4, activation='elu'),
    net.Dense(3, activation='elu'),
    net.Dense(2, activation='elu'),
  )

  model.compile(
    optimizer='none', 
    loss='mean squared error', 
    learning_rate=0.01,
    batch_size=1,
    epochs=50000, 
    metrics=['accuracy'],
  )
  
  model.fit(
    training_features, 
    training_target, 
  )
  
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

  model = net.Sequential(
    net.Recurrent('none', input=True, output=False),
    net.Recurrent('none', input=True, output=False),
    net.Recurrent('none', input=True, output=False),
    net.Recurrent('none', input=True, output=False),
    net.Recurrent('none', input=False, output=True),
  )
  
  model.compile(
    optimizer='rmsprop', 
    loss='mean squared error', 
    learning_rate=0.001,
    batch_size=3, 
    epochs=10000,
    metrics=['accuracy']
  ) 
  
  model.fit(
    training_features, 
    training_target, 
    )
  
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

  model = net.Sequential(
    net.LSTM(input=True, output = False),
    net.LSTM(input=True, output = False),
    net.LSTM(input=True, output = False),
    net.LSTM(input=True, output = False),
    net.LSTM(input=False, output = True),
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

  model = net.Sequential(
    net.GRU(input=True, output = False),
    net.GRU(input=True, output = False),
    net.GRU(input=True, output = False),
    net.GRU(input=True, output = False),
    net.GRU(input=False, output = True),
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
    )
   
  for feature, target in zip(training_features, training_target):
    print(f"pred {model.push(feature)} true {target}")

elif test == 6: # Paralellization test
  
  training_features = [
    
    generate_random_array(28)
    
  ]

  training_target = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  ]
  
  model = net.Sequential( 
        
    net.Parallel(
      net.Dense(5, 'elu'),
      net.Dense(5, 'elu'),
      net.Dense(5, 'elu'),
      net.Dense(5, 'elu'),
    ),
    net.Parallel(
      net.Dense(5, 'elu'),
      net.Dense(5, 'elu'),
      net.Dense(5, 'elu'),
      net.Dense(5, 'elu'),
    ),
    net.Merge('concat'),
    net.Dense(128, 'elu'),
    net.Dense(10, 'none'),
    net.Operation('softmax')
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
  )
  
  for feature, target in zip(training_features, training_target):
    print(f"pred {model.push(feature)} true {target}")

elif test == 7: # Early Stopping and what not

  def dataset(num_samples, input_dimensions, output_dimensions, noise_strength):
    """
    Generates synthetic noisy data for linear regression using pure Python lists.

    Args:
        num_samples (int): The number of data points to generate.
        input_dimensions (int): The number of features for each input sample.
        output_dimensions (int): The number of target values for each output sample.
                                  (Default: 1 for simple linear regression)
        noise_strength (float): The magnitude of random noise added to the outputs.
                                  A higher value means more noise.

    Returns:
        tuple[list, list]: A tuple containing two lists:
                            - inputs (list of lists): [[x1, x2, ...], [x1, x2, ...], ...]
                            - outputs (list of lists): [[y1, y2, ...], [y1, y2, ...], ...]
    """
    inputs = []
    outputs = []

    # Generate random "true" weights and biases for the underlying linear relationship
    # These are the parameters your linear regression model would try to learn
    true_weights = [
      [random.uniform(-1, 1) for _ in range(output_dimensions)]
      for _ in range(input_dimensions)
    ]
    true_biases = [random.uniform(-0.5, 0.5) for _ in range(output_dimensions)]

    for _ in range(num_samples):
      # Generate a random input sample
      current_input = [random.uniform(-5, 5) for _ in range(input_dimensions)]
      inputs.append(current_input)

      # Calculate the true linear output before adding noise
      current_true_output = [0.0] * output_dimensions
      for out_dim in range(output_dimensions):
        dot_product = 0.0
        for in_dim in range(input_dimensions):
          dot_product += current_input[in_dim] * true_weights[in_dim][out_dim]
        current_true_output[out_dim] = dot_product + true_biases[out_dim]

      # Add noise to the output
      current_noisy_output = [
        val + random.uniform(-noise_strength, noise_strength)
        for val in current_true_output
      ]
      outputs.append(current_noisy_output)

    return inputs, outputs

  def binary_dataset(num_samples: int, input_dimensions: int, output_dimensions: int, noise_strength: float):
      """
      Generates synthetic noisy binary classification data suitable for logistic regression.

      The 'true' underlying relationship is linear in the inputs, transformed by a sigmoid,
      and then thresholded to binary (0 or 1) with noise influencing the decision boundary.

      Args:
          num_samples (int): The number of data points to generate.
          input_dimensions (int): The number of features for each input sample.
          output_dimensions (int): The number of binary target values for each output sample.
                                  (Typically 1 for standard binary classification,
                                    but can be >1 for multi-label binary classification where
                                    each output is an independent binary classification).
          noise_strength (float): The magnitude of random noise added to the *logit*
                                  (the linear score before the sigmoid). A higher value
                                  makes the classes less separable (more overlap).

      Returns:
          tuple[list, list]: A tuple containing two lists:
                            - inputs (list of lists): [[x1, x2, ...], [x1, x2, ...], ...]
                            - outputs (list of lists): [[0 or 1], [0 or 1], ...]
                              (Each inner list will have `output_dimensions` elements).
      """
      inputs = []
      outputs = []

      # Generate random "true" weights and biases for the underlying linear relationship
      # These parameters define the true decision boundary for each output dimension
      true_weights = [
          [random.uniform(-1, 1) for _ in range(output_dimensions)]
          for _ in range(input_dimensions)
      ]
      true_biases = [random.uniform(-0.5, 0.5) for _ in range(output_dimensions)]

      for _ in range(num_samples):
          # Generate a random input sample (features)
          current_input = [random.uniform(-5, 5) for _ in range(input_dimensions)]
          inputs.append(current_input)

          current_binary_output = [0] * output_dimensions # Initialize list for current sample's binary outputs
          
          for out_dim in range(output_dimensions):
              # Calculate the linear score (logit) for the current output dimension
              logit = 0.0
              for in_dim in range(input_dimensions):
                  logit += current_input[in_dim] * true_weights[in_dim][out_dim]
              logit += true_biases[out_dim]

              # Add noise to the logit. This makes the probability 'fuzzy' around the decision boundary.
              noisy_logit = logit + random.uniform(-noise_strength, noise_strength)

              # Apply the sigmoid function to convert the noisy logit into a probability (0 to 1)
              probability = 1.0 / (1.0 + math.exp(-noisy_logit))

              # Threshold the probability to get a binary class (0 or 1)
              # If the probability is 0.5 or higher, classify as 1; otherwise, 0.
              predicted_class = 1 if probability >= 0.5 else 0
              
              current_binary_output[out_dim] = predicted_class
          
          outputs.append(current_binary_output) # Add the list of binary outputs for this sample

      return inputs, outputs

  training_features, training_target = dataset(40, 1, 1, 0.99)
  
  model = net.Sequential(
    net.Dense(20, 'tandip', initialization='he normal'),
    net.Dense(50, 'tandip', initialization='he normal'),
    net.Dense(1, 'none', initialization='he normal'),
  )

  model.compile(
    optimizer='adam', 
    loss='mean squared error', 
    learning_rate=0.001,
    batch_size=1,
    epochs=20000, 
    metrics=['accuracy'],
    validation=True,
    split=0.8,
    early_stopping=True,
    patience=5,
    logging=100,
    verbose=4
  )

  model.fit(
    training_features,
    training_target
  )


end_time = time.perf_counter()
duration = end_time - start_time
print(f"""
      finished training in {duration} seconds
      """)

# visualization and diagnosing

testing_domain = [(x/10)-10 for x in range(200)]
predicted_values = [model.push([x]) for x in testing_domain]

plt.subplot(2, 1, 1)
plt.plot(range(len(model.error_logs))           , model.error_logs           , color='red')
plt.plot(range(len(model.validation_error_logs)), model.validation_error_logs, color='blue')

plt.subplot(2, 1, 2)
plt.scatter(training_features, training_target, color='black', s=20)
plt.plot(testing_domain, predicted_values, color='red')
plt.show()

# plt.plot(range(len(model.error_logs)), model.error_logs)
# plt.title("Model Loss vs Epoch")
# plt.xlabel("Epoch")
# plt.ylabel(f" {model.loss} Loss")
# plt.show()