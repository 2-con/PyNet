import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax.numpy as jnp
from api.netflash import Sequential
from core.flash.layers import *
from tools.visual import array_display
import time
import matplotlib.pyplot as plt
from core.flash.callback import Callback

# Example Usage
model = Sequential(
  Dense(10, 'identity'),
  Dense(6, 'prelu'),
  Dense(4, 'prelu'),
)

class RealTimePlotter(Callback):
  def __init__(callbackself):
    callbackself.fig = None
    callbackself.ax = None
    callbackself.lines = {}
      
  def initialization(callbackself, *args, **kwargs):
    plt.ion()  # Turn on interactive mode
    callbackself.fig, callbackself.ax = plt.subplots()
    callbackself.ax.set_xlabel('Epoch')
    callbackself.ax.set_ylabel('Loss')
    callbackself.ax.set_yscale('log')
    callbackself.ax.grid(True)
    
    # Create line objects for training and validation loss
    callbackself.lines['train'], = callbackself.ax.plot([], [])
    callbackself.lines['validation'], = callbackself.ax.plot([], [])
      
  def after_epoch(callbackself, *args, **kwargs):
        
    x_data = list(range(kwargs.get('epoch', 0) + 1))
    y_data_train = kwargs.get('self', None).error_logs
    y_data_validation = kwargs.get('self', None).validation_error_logs
    
    callbackself.lines['train'].set_data(x_data, y_data_train)
    callbackself.lines['validation'].set_data(x_data, y_data_validation)
    
    # Dynamically adjust plot limits
    callbackself.ax.set_xlim(0, max(x_data) + 1 if x_data else 1)
    
    callbackself.ax.set_ylim(min(min(y_data_train), min(y_data_validation)) - 0.1, max(max(y_data_train), max(y_data_validation)) + 0.1)
    
    # Draw and flush events to update the plot
    callbackself.fig.canvas.draw()
    callbackself.fig.canvas.flush_events()
      
  def end(*args, **kwargs):
    plt.ioff()
    plt.show()

# Compile the model
model.compile(
  input_shape=(10,),
  optimizer='adam',
  loss='mean squared error',
  learning_rate=0.001,
  epochs=1000,
  metrics=['accuracy'], 
  batch_size=1,
  verbose=6,
  logging=1,
  validation_split=0.3,
  callback = RealTimePlotter,
)

# some dummy data for training
features = jax.random.uniform(key=jax.random.key(random.randint(1,1000)), minval=0, maxval=10, shape=(20,10))
targets = jax.random.uniform(key=jax.random.key(random.randint(1,1000)), minval=0, maxval=10, shape=(20,4))
# jnp.array([[0,0],[0,1],[1,0],[1,1]])
# jnp.array([[0],[1],[1],[0]])

# Fit the model
start = time.perf_counter()

model.fit(features, targets) 

print(f"""
      Finished training in {time.perf_counter() - start} seconds
      """)