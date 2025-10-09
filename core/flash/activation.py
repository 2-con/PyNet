import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax.numpy as jnp
import jax
from jax import grad

class ActivationBase:
  """
  Base class for all activation functions.
  'parameters' is a list of parameter names (strings) that the activation uses.
  """
  parameters = [] 

# normalization
class Sigmoid(ActivationBase):
  @staticmethod
  def forward(x, *args, **kwargs):
    return 1.0 / (1.0 + jnp.exp(-x))
  
  @staticmethod
  def backward(error_out, x, *args, **kwargs):
    local_grad = (1.0 / (1.0 + jnp.exp(-x))) * (1 - (1.0 / (1.0 + jnp.exp(-x))))
    return {"x": error_out * local_grad} # Outputs dL/dz
class Tanh(ActivationBase):
  @staticmethod
  def forward(x, *args, **kwargs):
    return jnp.tanh(x)
  
  @staticmethod
  def backward(error_out, x, *args, **kwargs):
    local_grad = 1 - jnp.tanh(x)**2
    return {"x": error_out * local_grad} # Outputs dL/dz

class Binary_step(ActivationBase):
  @staticmethod
  def forward(x, *args, **kwargs):
    return jnp.where(x > 0, 1.0, 0.0)
  
  @staticmethod
  def backward(error_out, x, *args, **kwargs):
    return {"x": jnp.zeros_like(x)} # dL/dz is 0

class Softsign(ActivationBase):
  @staticmethod
  def forward(x, *args, **kwargs):
    return x / (1.0 + jnp.abs(x))
  
  @staticmethod
  def backward(error_out, x, *args, **kwargs):
    local_grad = 1 / (1 + jnp.abs(x))**2
    return {"x": error_out * local_grad} # Outputs dL/dz

class Softmax(ActivationBase):
  @staticmethod
  def forward(x, *args, **kwargs):
    exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)
  
  @staticmethod
  def backward(error_out, x, *args, **kwargs):
    s = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True)) / jnp.sum(jnp.exp(x - jnp.max(x, axis=-1, keepdims=True)), axis=-1, keepdims=True)
    local_grad = s * (1 - s) 
    return {"x": error_out * local_grad}

class ReLU(ActivationBase):
  @staticmethod
  def forward(x, *args, **kwargs):
    return jnp.maximum(0.0, x)
  
  @staticmethod
  def backward(error_out, x, *args, **kwargs):
    local_grad = jnp.where(x > 0, 1.0, 0.0)
    return {"x": error_out * local_grad} # Outputs dL/dz

class Softplus(ActivationBase):
  @staticmethod
  def forward(x, *args, **kwargs):
    return jnp.log(1.0 + jnp.exp(x))
  
  @staticmethod
  def backward(error_out, x, *args, **kwargs):
    local_grad = 1 / (1 + jnp.exp(-x))
    return {"x": error_out * local_grad} # Outputs dL/dz

class Mish(ActivationBase):
  @staticmethod
  def forward(x, *args, **kwargs):
    return x * jnp.tanh(jnp.log(1.0 + jnp.exp(x)))
  
  # Still requires JAX's grad because the derivative is complex
  @staticmethod
  def backward(error_out, x, *args, **kwargs):
    omega = lambda x : x * jnp.tanh(jnp.log(1.0 + jnp.exp(x)))
    local_grad = grad(omega)(x)
    return {"x": error_out * local_grad}

class Swish(ActivationBase):
  @staticmethod
  def forward(x, *args, **kwargs):
    return x * (1.0 / (1.0 + jnp.exp(-x)))
  
  @staticmethod
  def backward(error_out, x, *args, **kwargs):
    s = (1.0 / (1.0 + jnp.exp(-x))) # Sigmoid(x)
    s_prime = s * (1 - s)           # Sigmoid'(x)
    local_grad = s + x * s_prime    # d(x*s)/dx
    return {"x": error_out * local_grad}

class Leaky_ReLU(ActivationBase):
  @staticmethod
  def forward(x, *args, **kwargs):
    return jnp.maximum(0.1 * x, x)
  
  @staticmethod
  def backward(error_out, x, *args, **kwargs):
    local_grad = jnp.where(x > 0, 1.0, 0.1)
    return {"x": error_out * local_grad}

class GELU(ActivationBase):
  @staticmethod
  def forward(x, *args, **kwargs):
    return jax.nn.gelu(x)
  
  @staticmethod
  def backward(error_out, x, *args, **kwargs):
    local_grad = (jax.nn.gelu(x) + (x * (jnp.sqrt(2 / jnp.pi) * jnp.exp(-(x**2) / 2)))) # This is the derivative for the approximate formula.
    return {"x": error_out * local_grad}

class Linear(ActivationBase):
  @staticmethod
  def forward(x, *args, **kwargs):
    return x
  
  @staticmethod
  def backward(error_out, x, *args, **kwargs):
    return {"x": error_out * jnp.ones_like(x)}

class ReEU(ActivationBase):
  @staticmethod
  def forward(x, *args, **kwargs):
    conditions = [x > 10, x < -10]
    choices = [x, 0.0]
    return jnp.select(conditions, choices, default=jnp.minimum(jnp.exp(x), jnp.maximum(1.0, x + 1.0)))
  
  @staticmethod
  def backward(error_out, x, *args, **kwargs):
    conditions = [x > 10, x < -10]
    choices = [1.0, 0.0]
    local_grad = jnp.select(conditions, choices, default=jnp.minimum(jnp.exp(x), 1.0))
    
    return {"x": error_out * local_grad}

class ReTanh(ActivationBase):
  @staticmethod
  def forward(x, *args, **kwargs):
    return x * (jnp.tanh(x + 1.0) + 1.0) / 2.0
  
  @staticmethod
  def backward(error_out, x, *args, **kwargs):
    v = (jnp.tanh(x + 1.0) + 1.0) / 2.0
    v_prime = (1 - jnp.tanh(x + 1.0)**2) / 2.0
    
    local_grad = v + x * v_prime
    
    return {"x": error_out * local_grad}

##############################
#   parametric activations   #
##############################

class ELU(ActivationBase):
  parameters = ["alpha"]
  @staticmethod
  def forward(x, alpha, *args, **kwargs):
    # Simplified for demonstration; kept original logic as closely as possible
    return jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1.0))
  
  @staticmethod
  def backward(error_out, x, alpha, *args, **kwargs):
    local_grad_x = jnp.where(x > 0, 1.0, alpha * jnp.exp(x))
    local_grad_alpha = jnp.where(x <= 0, (jnp.exp(x) - 1.0), 0.0)
    
    return {
      "x": error_out * local_grad_x,
      "alpha": jnp.sum(error_out * local_grad_alpha)
    }

class SELU(ActivationBase):
  parameters = ["alpha", "beta"]
  # SELU parameters are typically fixed constants
  @staticmethod
  def forward(x, alpha, beta, *args, **kwargs):
    return beta * jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1.0))

  @staticmethod
  def backward(error_out, x, alpha, beta, *args, **kwargs):
    local_grad_x = beta * jnp.where(x > 0, 1.0, alpha * jnp.exp(x))
    local_grad_alpha = beta * jnp.where(x <= 0, (jnp.exp(x) - 1.0), 0.0)
    
    return {
      "x": error_out * local_grad_x,
      "alpha": jnp.sum(error_out * local_grad_alpha),
      "beta": jnp.sum(error_out * jnp.where(x > 0, x, (alpha * jnp.exp(x) - 1.0)))
    }

class PReLU(ActivationBase):
  parameters = ["alpha"]
  @staticmethod
  def forward(x, alpha, *args, **kwargs):
    return jnp.maximum(alpha * x, x)

  @staticmethod
  def backward(error_out, x, alpha, *args, **kwargs):
    local_grad_x = jnp.where(x > 0, 1.0, alpha)
    local_grad_alpha = jnp.where(x <= 0, x, 0.0)
    
    return {
      "x": error_out * local_grad_x,
      "alpha": jnp.sum(error_out * local_grad_alpha)
    }

class SiLU(ActivationBase):
  parameters = ["alpha"]
  @staticmethod
  def forward(x, alpha, *args, **kwargs):
    return x * (1.0 / (1.0 + jnp.exp(-alpha * x)))

  @staticmethod
  def backward(error_out, x, alpha, *args, **kwargs):
    s = (1.0 / (1.0 + jnp.exp(-alpha * x))) # Sigmoid(alpha * x)
    s_prime = s * (1 - s)                    # Sigmoid'(alpha * x)
    
    local_grad_x = s + x * alpha * s_prime
    local_grad_alpha = x**2 * s_prime 
    
    return {
      "x": error_out * local_grad_x,
      "alpha": jnp.sum(error_out * local_grad_alpha)
    }
