
import math
import pynet.tools.math as math2

class Optimizer:
  def __init__(self):
    self.storage_1 = {}
    self.storage_2 = {}

  def Default(lr, param, gradient, optimizer_self, **kwargs):
    # regular gradient descent
    return param - lr * gradient

  def Adam(lr, param, gradient, optimizer_self, **kwargs):
    # adaptive moment estimation

    alpha = 0.9 if kwargs.get('alpha') == None else kwargs.get('alpha')
    beta = 0.999 if kwargs.get('beta') == None else kwargs.get('beta')
    epsilon = 1e-8 if kwargs.get('epsilon') == None else kwargs.get('epsilon')
    param_id = kwargs.get('param_id')

    if param_id not in optimizer_self.storage_1: # M
      optimizer_self.storage_1[param_id] = 0

    if param_id not in optimizer_self.storage_2: # V
      optimizer_self.storage_2[param_id] = 0

    optimizer_self.storage_1[param_id] = (alpha * optimizer_self.storage_1[param_id]) + ((1 - alpha) * gradient)
    optimizer_self.storage_2[param_id] = (beta * optimizer_self.storage_2[param_id]) + ((1 - beta) * gradient**2)

    M_hat = optimizer_self.storage_1[param_id] / (1 - alpha)
    V_hat = optimizer_self.storage_2[param_id] / (1 - beta)

    return param - ((M_hat * lr) / (math.sqrt(V_hat) + epsilon)) 

  def RMSprop(lr, param, gradient, optimizer_self, **kwargs):
    # root mean square propagation

    alpha = 0.9 if kwargs.get('alpha') == None else kwargs.get('alpha')
    epsilon = 1e-8 if kwargs.get('epsilon') == None else kwargs.get('epsilon')
    param_id = kwargs.get('param_id')

    if param_id not in optimizer_self.storage_1: # this one to store all previous gradients
      optimizer_self.storage_1[param_id] = 0

    optimizer_self.storage_1[param_id] = (alpha * optimizer_self.storage_1[param_id]) + ((1 - alpha) * gradient**2)

    RMS_gradient = math.sqrt(optimizer_self.storage_1[param_id] + epsilon)

    return param - lr * (gradient / RMS_gradient)

  def Adagrad(lr, param, gradient, optimizer_self, **kwargs):
    # adaptive gradient

    epsilon = 1e-8 if kwargs.get('epsilon') == None else kwargs.get('epsilon')
    param_id = kwargs.get('param_id')

    if param_id not in optimizer_self.storage_1:

      optimizer_self.storage_1[param_id] = gradient**2
      return param - ( lr / (epsilon + math.sqrt(gradient**2)) ) * gradient
    else:

      optimizer_self.storage_1[param_id] += gradient**2
      return param - ( lr / (epsilon + math.sqrt(optimizer_self.storage_1[param_id])) ) * gradient

  def Adadelta(lr, param, gradient, optimizer_self, **kwargs):
    # adaptive delta

    alpha = 0.9 if kwargs.get('alpha') == None else kwargs.get('alpha')
    epsilon = 1e-8 if kwargs.get('epsilon') == None else kwargs.get('epsilon')
    param_id = kwargs.get('param_id')

    if param_id not in optimizer_self.storage_1: # this one to store all previous gradients
      optimizer_self.storage_1[param_id] = 0

    if param_id not in optimizer_self.storage_2: # this one to store all previous squared delta
      optimizer_self.storage_2[param_id] = 0

    optimizer_self.storage_1[param_id] = (alpha * optimizer_self.storage_1[param_id]) + ((1 - alpha) * gradient**2)

    RMS_gradient = math.sqrt(optimizer_self.storage_1[param_id] + epsilon)
    RMS_delta = math.sqrt(optimizer_self.storage_2[param_id] + epsilon)

    delta = (RMS_delta/RMS_gradient) * gradient

    optimizer_self.storage_2[param_id] = (alpha * optimizer_self.storage_2[param_id]) + ((1 - alpha) * delta**2)

    return param - delta

  def Adamax(lr, param, gradient, optimizer_self, **kwargs):
    # adaptive moment max

    alpha = 0.9 if kwargs.get('alpha') == None else kwargs.get('alpha')
    beta = 0.999 if kwargs.get('beta') == None else kwargs.get('beta')
    epsilon = 1e-8 if kwargs.get('epsilon') == None else kwargs.get('epsilon')
    param_id = kwargs.get('param_id')

    if param_id not in optimizer_self.storage_1: # M
      optimizer_self.storage_1[param_id] = 0

    if param_id not in optimizer_self.storage_2: # L infinity
      optimizer_self.storage_2[param_id] = 0

    optimizer_self.storage_1[param_id] = (alpha * optimizer_self.storage_1[param_id]) + ((1 - alpha) * gradient)
    optimizer_self.storage_2[param_id] = max(beta * optimizer_self.storage_2[param_id], abs(gradient))

    M_hat = optimizer_self.storage_1[param_id] / (1 - alpha)

    return param - (lr * M_hat/(optimizer_self.storage_2[param_id] + epsilon))

  def Amsgrad(lr, param, gradient, optimizer_self, **kwargs):
    # adaptive moment square gradient

    alpha = 0.9 if kwargs.get('alpha') == None else kwargs.get('alpha')
    beta = 0.999 if kwargs.get('beta') == None else kwargs.get('beta')
    epsilon = 1e-8 if kwargs.get('epsilon') == None else kwargs.get('epsilon')
    param_id = kwargs.get('param_id')

    if param_id not in optimizer_self.storage_1: # M
      optimizer_self.storage_1[param_id] = 0

    if param_id not in optimizer_self.storage_2: # V
      optimizer_self.storage_2[param_id] = 0

    optimizer_self.storage_1[param_id] = (alpha * optimizer_self.storage_1[param_id]) + ((1 - alpha) * gradient)
    optimizer_self.storage_2[param_id] = (beta * optimizer_self.storage_2[param_id]) + ((1 - beta) * gradient**2)

    M_hat = optimizer_self.storage_1[param_id] / (1 - alpha)
    V_hat = max(optimizer_self.storage_2[param_id] / (1 - beta),optimizer_self.storage_2[param_id])

    return param - (lr / (math.sqrt(V_hat) + epsilon)) * M_hat

  def Gradclip(lr, param, gradient, optimizer_self, **kwargs):

    alpha = -math.inf if kwargs.get('minimum') == None else kwargs.get('alpha')
    beta = math.inf if kwargs.get('maximum') == None else kwargs.get('maximum')

    return param - lr * math2.clamp(gradient, beta, alpha)

  def SGND(lr, param, gradient, optimizer_self, **kwargs):
    # sign gradient descent

    return param - lr * math2.sgn(gradient)

  def Variational_Momentum(lr, param, gradient, optimizer_self, **kwargs):
    # experimental optimizer

    alpha = 1.1 if kwargs.get('alpha') == None else kwargs.get('alpha')
    beta  = 0.9 if kwargs.get('beta') == None else kwargs.get('beta')

    param_id = kwargs.get('param_id')

    if param_id not in optimizer_self.storage_1: # stores latest gradient
      optimizer_self.storage_1[param_id] = gradient

    if param_id not in optimizer_self.storage_2: # stores spesific LR
      optimizer_self.storage_2[param_id] = lr

    if optimizer_self.storage_1[param_id] * gradient >= 0:
      new_LR = lr * alpha

    else:
      new_LR = lr * beta

    optimizer_self.storage_1[param_id] = gradient
    optimizer_self.storage_2[param_id] = new_LR

    return param - new_LR * gradient

  def Momentum(lr, param, gradient, optimizer_self, **kwargs):
    # momentum

    alpha = 0.9 if kwargs.get('alpha') == None else kwargs.get('alpha')
    param_id = kwargs.get('param_id')

    if param_id not in optimizer_self.storage_1:

      optimizer_self.storage_1[param_id] = lr * gradient
      return param - optimizer_self.storage_1[param_id]

    else:

      new_velocity = (alpha * optimizer_self.storage_1[param_id]) + (lr * gradient)
      optimizer_self.storage_1[param_id] = new_velocity
      return param - new_velocity
