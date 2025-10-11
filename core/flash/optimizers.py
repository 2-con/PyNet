import jax
import jax.numpy as jnp

class AMSgrad:
  def update(lr, param, gradient, opt_state, **kwargs):
    alpha = kwargs.get('alpha', 0.9)
    beta = kwargs.get('beta', 0.999)
    epsilon = kwargs.get('epsilon', 1e-8)
    timestep = kwargs.get('timestep', 1)

    # Access state components by index
    m = opt_state[0]          # m
    v = opt_state[1]          # v
    v_hat_max = opt_state[2]  # v_hat_max

    m_new = (alpha * m) + ((1 - alpha) * gradient)
    v_new = (beta * v) + ((1 - beta) * jnp.square(gradient))

    v_hat_max_new = jnp.maximum(v_hat_max, v_new)

    M_hat = m_new / (1 - alpha**timestep)
    
    new_param = param - (lr / (jnp.sqrt(v_hat_max_new) + epsilon)) * M_hat

    # Return updated state components as a tuple
    new_opt_state = (m_new, v_new, v_hat_max_new)
    return new_param, new_opt_state
  
  def initialize(param_shape, param_dtype):
    return (
        jnp.zeros(param_shape, dtype=param_dtype),  # m
        jnp.zeros(param_shape, dtype=param_dtype),  # v
        jnp.zeros(param_shape, dtype=param_dtype)   # v_hat_max
    )

class Default:
  def update(lr, param, gradient, opt_state, **kwargs):
    new_param = param - lr * gradient
    return new_param, opt_state
  
  def initialize(param_shape, param_dtype):
    return ()

class Gradclip:
  def update(lr, param, gradient, opt_state, **kwargs):
    minimum = kwargs.get('minimum', -1e-4)
    maximum = kwargs.get('maximum', 1e-4)

    new_param = param - lr * jnp.clip(gradient, minimum, maximum)
    return new_param, opt_state # opt_state (empty tuple/list) just passed through
  
  def initialize(param_shape, param_dtype):
    return ()

class SGND:
  def update(lr, param, gradient, opt_state, **kwargs):
    new_param = param - lr * jnp.sign(gradient)
    return new_param, opt_state # opt_state (empty tuple/list) just passed through
  
  def initialize(param_shape, param_dtype):
    return ()

class Momentum:
  def update(lr, param, gradient, opt_state, **kwargs):
    alpha = kwargs.get('alpha', 0.9)
    # Access state components by index
    velocity = opt_state[0] # velocity
    
    new_velocity = (alpha * velocity) + (lr * gradient)
    new_param = param - new_velocity
    
    # Return updated state components as a tuple
    new_opt_state = (new_velocity,) # Tuple with one element
    return new_param, new_opt_state
  
  def initialize(param_shape, param_dtype):
    return jnp.zeros(param_shape, dtype=param_dtype),  # velocity
  
class RMSprop:
  def update(lr, param, gradient, opt_state, **kwargs):
    alpha = kwargs.get('alpha', 0.9)
    epsilon = kwargs.get('epsilon', 1e-8)
    # Access state components by index
    avg_sq_grad = opt_state[0] # avg_sq_grad
    
    avg_sq_grad_new = (alpha * avg_sq_grad) + ((1 - alpha) * jnp.square(gradient))
    RMS_gradient = jnp.sqrt(avg_sq_grad_new + epsilon)
    new_param = param - lr * (gradient / RMS_gradient)
    
    # Return updated state components as a tuple
    new_opt_state = (avg_sq_grad_new,) # Tuple with one element
    return new_param, new_opt_state
  
  def initialize(param_shape, param_dtype):
    return jnp.zeros(param_shape, dtype=param_dtype),  # avg_sq_grad

class Adagrad:
  def update(lr, param, gradient, opt_state, **kwargs):
    epsilon = kwargs.get('epsilon', 1e-8)
    # Access state components by index
    sum_sq_grad = opt_state[0] # sum_sq_grad
    
    sum_sq_grad_new = sum_sq_grad + jnp.square(gradient)
    new_param = param - (lr / (jnp.sqrt(sum_sq_grad_new) + epsilon)) * gradient
    
    # Return updated state components as a tuple
    new_opt_state = (sum_sq_grad_new,) # Tuple with one element
    return new_param, new_opt_state
  
  def initialize(param_shape, param_dtype):
    return jnp.zeros(param_shape, dtype=param_dtype),  # sum_sq_grad

class Novograd:
  def update(lr, param, gradient, opt_state, **kwargs):
    alpha = kwargs.get('alpha', 0.9)
    beta = kwargs.get('beta', 0.999)
    epsilon = kwargs.get('epsilon', 1e-8)
    timestep = kwargs.get('timestep', 1)
    # Access state components by index
    m = opt_state[0] # m
    v = opt_state[1] # v
    
    normalized_gradient = gradient / (jnp.abs(gradient) + epsilon)
    m_new = (alpha * m) + ((1 - alpha) * normalized_gradient)
    v_new = (beta * v) + ((1 - beta) * jnp.square(gradient))
    
    M_hat = m_new / (1 - alpha**timestep)
    V_hat = v_new / (1 - beta**timestep)
    new_param = param - ((M_hat * lr) / (jnp.sqrt(V_hat) + epsilon))
    
    # Return updated state components as a tuple
    new_opt_state = (m_new, v_new)
    return new_param, new_opt_state
  
  def initialize(param_shape, param_dtype):
    return (
        jnp.zeros(param_shape, dtype=param_dtype),  # m
        jnp.zeros(param_shape, dtype=param_dtype)   # v
    )

class Adam:
  def update(lr, param, gradient, opt_state, **kwargs):
    alpha = kwargs.get('alpha', 0.9)
    beta = kwargs.get('beta', 0.999)
    epsilon = kwargs.get('epsilon', 1e-8)
    timestep = kwargs.get('timestep', 1)
    # Access state components by index
    m = opt_state[0] # m
    v = opt_state[1] # v
    
    m_new = (alpha * m) + ((1 - alpha) * gradient)
    v_new = (beta * v) + ((1 - beta) * jnp.square(gradient))
    
    M_hat = m_new / (1 - alpha**timestep)
    V_hat = v_new / (1 - beta**timestep)
    new_param = param - ((M_hat * lr) / (jnp.sqrt(V_hat) + epsilon))
    
    # Return updated state components as a tuple
    new_opt_state = (m_new, v_new)
    return new_param, new_opt_state
  
  def initialize(param_shape, param_dtype):
    return (
        jnp.zeros(param_shape, dtype=param_dtype),  # m
        jnp.zeros(param_shape, dtype=param_dtype)   # v
    )

class Adadelta:
  def update(lr, param, gradient, opt_state, **kwargs):
    alpha = kwargs.get('alpha', 0.9)
    epsilon = kwargs.get('epsilon', 1e-8)
    # Access state components by index
    avg_sq_grad = opt_state[0] # avg_sq_grad
    avg_sq_delta = opt_state[1] # avg_sq_delta
    
    avg_sq_grad_new = (alpha * avg_sq_grad) + ((1 - alpha) * jnp.square(gradient))
    RMS_gradient = jnp.sqrt(avg_sq_grad_new + epsilon)
    RMS_delta = jnp.sqrt(avg_sq_delta + epsilon)
    
    delta = (RMS_delta / RMS_gradient) * gradient
    avg_sq_delta_new = (alpha * avg_sq_delta) + ((1 - alpha) * jnp.square(delta))
    new_param = param - delta
    
    # Return updated state components as a tuple
    new_opt_state = (avg_sq_grad_new, avg_sq_delta_new)
    return new_param, new_opt_state

  def initialize(param_shape, param_dtype):
    return (
        jnp.zeros(param_shape, dtype=param_dtype),  # avg_sq_grad
        jnp.zeros(param_shape, dtype=param_dtype)   # avg_sq_delta
    )

class Adamax:
  def update(lr, param, gradient, opt_state, **kwargs):
    alpha = kwargs.get('alpha', 0.9)
    beta = kwargs.get('beta', 0.999)
    epsilon = kwargs.get('epsilon', 1e-8)
    # Access state components by index
    m = opt_state[0] # m
    u_inf = opt_state[1] # u_inf
    
    m_new = (alpha * m) + ((1 - alpha) * gradient)
    u_inf_new = jnp.maximum(beta * u_inf, jnp.abs(gradient))
    M_hat = m_new / (1 - alpha) # No timestep needed for bias correction in Adamax m
    new_param = param - (lr * M_hat / (u_inf_new + epsilon))
    
    # Return updated state components as a tuple
    new_opt_state = (m_new, u_inf_new)
    return new_param, new_opt_state

  def initialize(param_shape, param_dtype):
    return (
        jnp.zeros(param_shape, dtype=param_dtype),  # m
        jnp.zeros(param_shape, dtype=param_dtype)   # u_inf (max of past gradients)
    )

class Rprop:
  def update(lr, param, gradient, opt_state, **kwargs):
    alpha = kwargs.get('alpha', 1.1) # Increase factor
    beta = kwargs.get('beta', 0.5)  # Decrease factor
    min_step = kwargs.get('min_step', 1e-6) # Minimum step size
    max_step = kwargs.get('max_step', 50.0) # Maximum step size
    
    # Access state components by index
    prev_grad = opt_state[0] # prev_grad
    step_size = opt_state[1] # step_size

    signs_agree = jnp.sign(prev_grad) * jnp.sign(gradient) > 0.0
    
    # Increase or decrease step size
    new_step_size = jnp.where(
      signs_agree,
      step_size * alpha,
      step_size * beta
    )
    
    # Clip step size
    new_step_size = jnp.clip(new_step_size, min_step, max_step)
    
    # If signs disagree, the previous gradient is set to zero (for next iteration)
    # This prevents oscillations
    new_prev_grad = jnp.where(signs_agree, gradient, jnp.zeros_like(gradient))
    
    # Calculate update delta
    update_delta = jnp.where(signs_agree, new_step_size * jnp.sign(gradient), jnp.zeros_like(gradient))
    
    new_param = param - update_delta
    
    # Return updated state components as a tuple
    new_opt_state = (new_prev_grad, new_step_size)
    return new_param, new_opt_state

  def initialize(param_shape, param_dtype):
    return (
        jnp.zeros(param_shape, dtype=param_dtype),      # prev_grad
        jnp.full(param_shape, 0.01, dtype=param_dtype)  # step_size (often initialized to a small constant)
    )
