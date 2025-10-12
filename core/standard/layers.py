import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abc import ABC, abstractmethod

class Layer(ABC):
  """
  Base class for all layers
  
  Considering that layers can have differing operations, make sure the internal components agree with the NetLab API which means
  a functional-programming paradigm and correct capitalization (class methods are always lowercase). Attributes of the layer should stay constant and immutable.
  
  A Layer class is required to have the following:
  - '__init__' : method with any constant object attributes should be defined here
    - Args:
      - Any
    - Returns:
      - None
    
  - 'calibrate' : method that will be called once per layer during compilation to generate a weight matrix as well as constant object attributes
    - Args:
      - fan_in tuple[int, ...] : shape of the input to the layer
      - fan_out_shape tuple[int, ...] : shape of the output of the layer
    - Returns:
      - dict : weight_matrix (including bias and parametric values) for the whole layer
      - tuple[int, ...] : shape of the output of the layer
      
  - 'forward' : method that will be called everytime the layer is called
    - Args:
      - params (dict) : weight_matrix (including bias and parametric values) for the whole layer
      - inputs (jnp.ndarray) : the input to the layer
    - Returns:
      - jnp.ndarray : the output of the layer
      - jnp.ndarray : cached values to be used in the backward pass
      
  - 'backward' : method that will be called before the 'update' method
    - Args:
      - params (dict) : weight_matrix (including bias and parametric values) for the whole layer
      - inputs (jnp.ndarray) : the input to the layer
      - error (jnp.ndarray) : the incoming error for the layer
      - weighted_sums (jnp.ndarray) : cached values from the forward pass
    - Returns:
      - jnp.ndarray : the upstream error to be passed to the next layer
      - dict : weight_matrix (including bias and parametric values) for the whole layer
  
  - 'update' : method that will be called if provided, otherwise NetLab/NetFlash will ignore the layer
    - Args:
      - optimizer_fn (optimizer.forward method) : this could be any callable, but make sure it adheres to the structure provided under lab.optimizer
      - learning_rate (float or jnp.float32) : learning rate
      - layer_params (dict) : weight_matrix (including bias and parametric values) for the whole layer
      - gradients (dict) : error gradients for the whole layer
      - opt_state (dict) : optimizer state for the whole layer
      - **other_parameters (kwargs) : any other optimizer hyperparameters that could be passed in
    - Returns:
      - dict : updated params for the layer
      - dict : updated opt state
  """
  @abstractmethod
  def __init__(self):
    """
    A Layer class is required to have the following:
    
    - A '__init__' method with any constant object attributes should be defined here
      - Args:
        - Any
      - Returns:
        - None
    """
    pass
  
  @abstractmethod
  def calibrate(self, fan_in:tuple[int, ...], fan_out_shape:int) -> tuple[dict, tuple[int, ...]]:
    """
    - A 'calibrate' method that will be called once per layer during compilation to generate a weight matrix as well as constant object attributes
      - Args:
        - fan_in tuple[int, ...] : shape of the input to the layer
        - fan_out_shape tuple[int, ...] : shape of the output of the layer
      - Returns:
        - dict : weight_matrix (including bias and parametric values) for the whole layer
        - tuple[int, ...] : shape of the output of the layer
    """
    pass
  
  @abstractmethod
  def forward(self):
    """
    - A 'forward' method that will be called everytime the layer is called
      - Args:
        - params (dict) : weight_matrix (including bias and parametric values) for the whole layer
        - inputs (jnp.ndarray) : the input to the layer
      - Returns:
        - jnp.ndarray : the output of the layer
        - jnp.ndarray : cached values to be used in the backward pass
    """
    pass
  
  @abstractmethod
  def backward(self):
    """
    - A 'backward' method
      - Args:
        - params (dict) : weight_matrix (including bias and parametric values) for the whole layer
        - inputs (jnp.ndarray) : the input to the layer
        - error (jnp.ndarray) : the incoming error for the layer
        - weighted_sums (jnp.ndarray) : cached values from the forward pass
      - Returns:
        - jnp.ndarray : the upstream error to be passed to the next layer
        - dict : weight_matrix (including bias and parametric values) for the whole layer
    """
    pass
  
  