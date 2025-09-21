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
  Dense(10, 'elu'),
  Dense(6, 'elu'),
  Dense(1, 'elu'),
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
    # callbackself.ax.set_yscale('log')
    callbackself.ax.grid(True)
    
    # Create line objects for training and validation loss
    callbackself.lines['train'], = callbackself.ax.plot([], [])
      
  def after_epoch(callbackself, *args, **kwargs):
        
    x_data = list(range(kwargs.get('epoch', 0) + 1))
    y_data_train = kwargs.get('self', None).error_logs
    
    callbackself.lines['train'].set_data(x_data, y_data_train)
    
    # Dynamically adjust plot limits
    callbackself.ax.set_xlim(0, max(x_data) + 1 if x_data else 1)
    
    callbackself.ax.set_ylim(min(y_data_train) - 0.1, max(y_data_train) + 0.1)
    
    # Draw and flush events to update the plot
    callbackself.fig.canvas.draw()
    callbackself.fig.canvas.flush_events()
      
  def end(*args, **kwargs):
    plt.ioff()
    plt.show()


# Compile the model
model.compile(
  input_shape=(2,),
  optimizer='adam',
  loss='mean squared error',
  learning_rate=0.001,
  epochs=1000,
  metrics=['accuracy'], 
  batch_size=1,
  verbose=4,
  logging=1,
  callback = RealTimePlotter,
)

# some dummy data for training
features = jnp.array([[0,0],[0,1],[1,0],[1,1]])
jax.random.uniform(key=jax.random.key(random.randint(1,1000)), minval=0, maxval=10, shape=(20,10))
targets = jnp.array([[0],[1],[1],[0]])
jax.random.uniform(key=jax.random.key(random.randint(1,1000)), minval=0, maxval=10, shape=(20,4))

# Fit the model
start = time.perf_counter()

model.fit(features, targets) 

print(f"""
      Finished training in {time.perf_counter() - start} seconds
      """)