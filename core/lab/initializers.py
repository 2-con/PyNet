import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abc import ABC, abstractmethod

class Initializer(ABC):
  """
  Base class for all initializers
  
  An initializer class is required to have a __call__ method that accepts a shape, fan_in and fan_out_size.
  the initializer is called once per layer inside a layer's calibrate() method
  """
  @abstractmethod
  def __call__(self, shape: tuple, fan_in: int, fan_out_size: int):
    """
    Main method: generates a weight matrix according to the inputs and a specified shape
    
    Args:
      shape (tuple): shape of the weight matrix
      fan_in (int): number of incoming connections
      fan_out_size (int): number of outgoing connections
      
    Returns:
      jnp.ndarray: weight matrix as specified in 'shape'
    """
    pass

