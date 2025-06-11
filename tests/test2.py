import sys
import os

# Get the directory of the current script (test.py)
current_script_dir = os.path.dirname(__file__)

# Navigate up one level to the 'PyNet' directory
# If test.py is in PyNet/tests/, then '..' takes us to PyNet/
pynet_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))

# Add the PyNet root directory to Python's module search path
sys.path.append(pynet_root_dir)

from tools import arraytools
from tools.visual import numerical_display
from tools.utility import timer

import numpy as np
from api.synapse import Dense
import jax.numpy as jnp
import jax

onedim  = [1,2,3,4,5]
onedim2 = [4,5,6]

twodim = [
  [1,2,3],
  [4,5,6],
  [7,8,9]
  ]

bar = [
  [1,2,3],
  [4,5,6],
  ]

biggertwodim = [
  [ 1, 2, 3, 4],
  [ 5, 6, 7, 8],
  [ 9,10,11,12],
  [13,14,15,16]
  ]

threedim = [
  [
    [ 1, 2, 3, 4],
    [ 5, 6, 7, 8],
    [ 9,10,11,12],
    [13,14,15,16]
  ],
  [
    [ 1, 2, 3, 4],
    [ 5, 6, 7, 8],
    [ 9,10,11,12],
    [13,14,15,16]
  ]
]

STRESS = 1000
inputsize = 100
outputsize = 100
test = arraytools.generate_random_array(inputsize)

@timer
def mydense(input, inputshape, outputshape):
  layer = Dense(outputshape, 'none')
  layer.reshape_input_shape(inputshape, (1,1))
  ans = []
  
  for _ in range(STRESS):
    ans = layer.apply(input)
  return ans

@timer
@jax.jit
def jaxdense(input, inputshape, outputshape):
  weights = jnp.array(arraytools.generate_random_array(100, 100))
  input_arr = jnp.array(input)
  ans = []
  
  for _ in range(STRESS):
    ans = jnp.matmul(input_arr, weights)
    
  return ans

@timer
def NPD(input, inputshape, outputshape):
  weights = np.array(arraytools.generate_random_array(100, 100))
  input_arr = np.array(input)
  ans = []
  
  for _ in range(STRESS):
    ans = np.matmul(input_arr, weights)
    
  return ans


print(f"""
        JAX: {jaxdense(test, inputsize, outputsize)}
      PyNet: {mydense(test, inputsize, outputsize)}
        NPD: {NPD(test, inputsize, outputsize)}
      
      """)