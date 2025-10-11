import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax.numpy as jnp
from abc import ABC, abstractmethod
from core.flash.activations import Sigmoid, Tanh, Linear, ReLU, Leaky_ReLU, PReLU # include widely-used functions only

class Activation(ABC):
  """
  Base class for all activation functions. 
  
  An activation function class is required to have the following:
  - A `forward` method for applying the activation function.
  - A `backward` method for computing the gradient of the activation function. 
  
  Attributes:
    parameters (list): A list of strings, where each string is the name of a parameter 
                       required by a parametric function. Defaults to an empty list for non-parametric activations.
  """
  parameters = []
  
  @abstractmethod
  def forward(x, *args, **kwargs):
    """
    Forward propagation method: Applies the activation function to the input.
    
    Args:
      x (jnp.ndarray): The input array to the activation function.
      *args: Variable length argument list.  Can be used to pass additional information to the activation function.
      **kwargs: Arbitrary keyword arguments. Used to pass parameters (if any) to the activation function. 
                Make sure the parameter names match those listed in the 'parameters' attribute.
    
    Returns:
      jnp.ndarray: The output array after applying the activation function, with the same dimensions as the input.
    """
    pass
  
  def backward(incoming_error, x, *args, **kwargs):
    """
    Backward propagation method: Computes the gradient of the activation function with respect to its input.
    
    Args:
      incoming_error (jnp.ndarray): The incoming error signal from the subsequent layer.
      x (jnp.ndarray): The input to the activation function during the forward pass.  This is needed to compute the gradient.
      *args: Variable length argument list.  Can be used to pass additional information to the activation function.
      **kwargs: Arbitrary keyword arguments. Used to pass parameters (if any) to the activation function.
    
    Returns:
      dict: A dictionary containing the gradient of the loss with respect to the key (incoming_error * local_gradient).  
            The key are 'x' along with any parametric parameters specified in 'parameters'.
    """
    pass
