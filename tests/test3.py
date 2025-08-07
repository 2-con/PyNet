import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api.alpha as net
import time
from tools.arraytools import generate_random_array
from core.vanilla.activation import ReLU
from core.vanilla.loss import Mean_squared_error as mse
from core.vanilla.derivative import ReLU_derivative
from tools.utility import progress_bar

features = generate_random_array(784,100)
targets = generate_random_array(10,100)

model = net.initialize(784, 64, 64, 10)

epochs = 10
learning_rate = 0.1

for epoch in range(epochs):
  for data_index in progress_bar(range(len(features)), "> Processing Batch ", f"Epoch {epoch}/{epochs} ({epoch/epochs*100:.2f})%"):
    feature = features[data_index]
    target = targets[data_index]
    activations, weighted_sums = net.propagate(model, feature, ReLU)
    error = net.backpropegate(model, activations, weighted_sums, target, ReLU_derivative)
    
    net.update(model, activations, error, learning_rate)

loss = 0

# evaluation
for feature, target in zip(features, targets):
  activations, weighted_sums = net.propagate(model, feature, ReLU)
  loss += mse(target, activations[-1])
  
print(f"Loss: {loss/len(features)}")