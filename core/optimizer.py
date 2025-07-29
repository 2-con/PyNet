import math
from tools.math import clamp, sgn

def Default(lr, param, gradient, storage_1, storage_2, **kwargs):
  # regular gradient descent
  return param - lr * gradient

def Novograd(lr, param, gradient, storage_1, storage_2, **kwargs):
  # adaptive moment estimation

  alpha = 0.9 if kwargs.get('alpha') is None else kwargs.get('alpha')
  beta = 0.999 if kwargs.get('beta') is None else kwargs.get('beta')
  epsilon = 1e-8 if kwargs.get('epsilon') is None else kwargs.get('epsilon')
  timestep = kwargs.get('timestep', 1)
  param_id = kwargs.get('param_id')

  if param_id not in storage_1: # M
    storage_1[param_id] = 0

  if param_id not in storage_2: # V
    storage_2[param_id] = 0

  normalized_gradient = gradient / (abs(gradient) + epsilon)
  
  storage_1[param_id] = (alpha * storage_1[param_id]) + ((1 - alpha) * normalized_gradient)
  storage_2[param_id] = (beta * storage_2[param_id]) + ((1 - beta) * gradient**2)

  M_hat = storage_1[param_id] / (1 - alpha**timestep)
  V_hat = storage_2[param_id] / (1 - beta**timestep)

  return param - ((M_hat * lr) / (math.sqrt(V_hat) + epsilon)) 

def Adam(lr, param, gradient, storage_1, storage_2, **kwargs):
  # adaptive moment estimation

  alpha = 0.9 if kwargs.get('alpha') is None else kwargs.get('alpha')
  beta = 0.999 if kwargs.get('beta') is None else kwargs.get('beta')
  epsilon = 1e-8 if kwargs.get('epsilon') is None else kwargs.get('epsilon')
  timestep = kwargs.get('timestep', 1)
  param_id = kwargs.get('param_id')

  if param_id not in storage_1: # M
    storage_1[param_id] = 0

  if param_id not in storage_2: # V
    storage_2[param_id] = 0

  storage_1[param_id] = (alpha * storage_1[param_id]) + ((1 - alpha) * gradient)
  storage_2[param_id] = (beta * storage_2[param_id]) + ((1 - beta) * gradient**2)

  M_hat = storage_1[param_id] / (1 - alpha**timestep)
  V_hat = storage_2[param_id] / (1 - beta**timestep)

  return param - ((M_hat * lr) / (math.sqrt(V_hat) + epsilon)) 

def RMSprop(lr, param, gradient, storage_1, storage_2, **kwargs):
  # root mean square propagation

  alpha = 0.9 if kwargs.get('alpha') is None else kwargs.get('alpha')
  epsilon = 1e-8 if kwargs.get('epsilon') is None else kwargs.get('epsilon')
  param_id = kwargs.get('param_id')

  if param_id not in storage_1: # this one to store all previous gradients
    storage_1[param_id] = 0

  storage_1[param_id] = (alpha * storage_1[param_id]) + ((1 - alpha) * gradient**2)

  RMS_gradient = math.sqrt(storage_1[param_id] + epsilon)

  return param - lr * (gradient / RMS_gradient)

def Adagrad(lr, param, gradient, storage_1, storage_2, **kwargs):
  # adaptive gradient

  epsilon = 1e-8 if kwargs.get('epsilon') is None else kwargs.get('epsilon')
  param_id = kwargs.get('param_id')

  if param_id not in storage_1:
    storage_1[param_id] = gradient**2
    return param - ( lr / (epsilon + math.sqrt(gradient**2)) ) * gradient
  else:
    storage_1[param_id] += gradient**2
    return param - ( lr / (epsilon + math.sqrt(storage_1[param_id])) ) * gradient

def Adadelta(lr, param, gradient, storage_1, storage_2, **kwargs):
  # adaptive delta

  alpha = 0.9 if kwargs.get('alpha') is None else kwargs.get('alpha')
  epsilon = 1e-8 if kwargs.get('epsilon') is None else kwargs.get('epsilon')
  param_id = kwargs.get('param_id')

  if param_id not in storage_1: # this one to store all previous gradients
    storage_1[param_id] = 0

  if param_id not in storage_2: # this one to store all previous squared delta
    storage_2[param_id] = 0

  storage_1[param_id] = (alpha * storage_1[param_id]) + ((1 - alpha) * gradient**2)

  RMS_gradient = math.sqrt(storage_1[param_id] + epsilon)
  RMS_delta = math.sqrt(storage_2[param_id] + epsilon)

  delta = (RMS_delta/RMS_gradient) * gradient

  storage_2[param_id] = (alpha * storage_2[param_id]) + ((1 - alpha) * delta**2)

  return param - delta

def Adamax(lr, param, gradient, storage_1, storage_2, **kwargs):
  # adaptive moment max

  alpha = 0.9 if kwargs.get('alpha') is None else kwargs.get('alpha')
  beta = 0.999 if kwargs.get('beta') is None else kwargs.get('beta')
  epsilon = 1e-8 if kwargs.get('epsilon') is None else kwargs.get('epsilon')
  param_id = kwargs.get('param_id')

  if param_id not in storage_1: # M
    storage_1[param_id] = 0

  if param_id not in storage_2: # L infinity
    storage_2[param_id] = 0

  storage_1[param_id] = (alpha * storage_1[param_id]) + ((1 - alpha) * gradient)
  storage_2[param_id] = max(beta * storage_2[param_id], abs(gradient))

  M_hat = storage_1[param_id] / (1 - alpha)

  return param - (lr * M_hat/(storage_2[param_id] + epsilon))

def Amsgrad(lr, param, gradient, storage_1, storage_2, **kwargs):
  # adaptive moment square gradient

  alpha = 0.9 if kwargs.get('alpha') is None else kwargs.get('alpha')
  beta = 0.999 if kwargs.get('beta') is None else kwargs.get('beta')
  epsilon = 1e-8 if kwargs.get('epsilon') is None else kwargs.get('epsilon')
  param_id = kwargs.get('param_id')

  if param_id not in storage_1: # M
    storage_1[param_id] = 0

  if param_id not in storage_2: # V
    storage_2[param_id] = 0

  storage_1[param_id] = (alpha * storage_1[param_id]) + ((1 - alpha) * gradient)
  storage_2[param_id] = (beta * storage_2[param_id]) + ((1 - beta) * gradient**2)

  M_hat = storage_1[param_id] / (1 - alpha)
  V_hat = max(storage_2[param_id] / (1 - beta), storage_2[param_id])

  return param - (lr / (math.sqrt(V_hat) + epsilon)) * M_hat

def Gradclip(lr, param, gradient, storage_1, storage_2, **kwargs):
  minimum = -1e-4 if kwargs.get('alpha') is None else kwargs.get('alpha')
  maximum = 1e-4 if kwargs.get('beta') is None else kwargs.get('beta')

  return param - lr * clamp(gradient, minimum, maximum)

def SGND(lr, param, gradient, storage_1, storage_2, **kwargs):
  # sign gradient descent

  return param - lr * sgn(gradient)

def Rprop(lr, param, gradient, storage_1, storage_2, **kwargs):
  """
  by far the goofiest optimizer
  """
  
  alpha = 1.1 if kwargs.get('alpha') is None else kwargs.get('alpha') # reward
  beta = 0.9 if kwargs.get('beta') is None else kwargs.get('beta') # punishment
  gamma = 0.5 if kwargs.get('gamma') is None else kwargs.get('gamma') # max step size
  epsilon = 1e-5 if kwargs.get('epsilon') is None else kwargs.get('epsilon') # min step size

  param_id = kwargs.get('param_id')

  if param_id not in storage_1: # stores latest gradient
    storage_1[param_id] = gradient

  if param_id not in storage_2: # stores latest gradient
    storage_2[param_id] = lr # Stores step size

  # Check sign agreement
  if sgn(storage_1[param_id]) == sgn(gradient):
    new_LR = storage_2[param_id] * alpha
    # Update stored gradient only if signs agree
    storage_1[param_id] = gradient 
  else:
    # If signs disagree, the gradient for this step is effectively zero, and the previous gradient is "reset"
    new_LR = storage_2[param_id] * beta
    gradient = 0 # This ensures no update for this step if signs flip
    storage_1[param_id] = 0 # Reset stored gradient to 0 or previous gradient (depends on Rprop variant)

  new_LR = clamp(new_LR, epsilon, gamma)
  
  storage_2[param_id] = new_LR 
  
  return param - new_LR * sgn(gradient)

def Momentum(lr, param, gradient, storage_1, storage_2, **kwargs):
  # momentum

  alpha = 0.9 if kwargs.get('alpha') is None else kwargs.get('alpha')
  param_id = kwargs.get('param_id')

  if param_id not in storage_1: # velocity storage
    storage_1[param_id] = lr * gradient # Initialize velocity
    return param - storage_1[param_id]
  else:
    new_velocity = (alpha * storage_1[param_id]) + (lr * gradient)
    storage_1[param_id] = new_velocity
    return param - new_velocity