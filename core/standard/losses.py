import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax.numpy as jnp
from abc import ABC, abstractmethod


class Loss(ABC):
  """
  Base class for all Loss functions
  
  A Loss class must implement the following methods:
  - 'forward' : the forward pass of the loss function discounting regularization
    - Args:
      - y_true (jnp.ndarray): the true labels for the batch
      - y_pred (jnp.ndarray): the predicted labels for the batch
    - Returns:
      - jnp.float32: the loss value
  
  - 'backward' : the backward pass of the loss function for the initial error
    - Args:
      - y_true (jnp.ndarray): the true labels for the batch
      - y_pred (jnp.ndarray): the predicted labels for the batch
    - Returns:
      - jnp.ndarray: the initial error that will be fed into the backpropagation process
  """
  @abstractmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    The forward pass of the loss function are used to compute the loss, this internal method should be made a static method
    since the loss function is stateless
    
    - 'forward' : the forward pass of the loss function discounting regularization
    - Args:
      - y_true (jnp.ndarray): the true labels for the batch
      - y_pred (jnp.ndarray): the predicted labels for the batch
    - Returns:
      - jnp.float32: the loss value
    """
    pass

  @abstractmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    The backward pass of the loss function are used to compute the initial error, this internal method should be made a static method
    since the loss function is stateless. StandardNet/NetLab does not check for NaN or mathematical errors.
    
    - 'backward' : the backward pass of the loss function for the initial error
    - Args:
      - y_true (jnp.ndarray): the true labels for the batch
      - y_pred (jnp.ndarray): the predicted labels for the batch
    - Returns:
      - jnp.ndarray: the initial error that will be fed into the backpropagation process
    """
    pass
