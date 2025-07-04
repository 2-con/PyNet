import math
import core.activation as Activation

# scalers

def Standard_scaler_derivative(x, std):
  return 1 / std if std != 0 else 0

def Min_max_scaler_derivative(x, min, max):
  return 1 / (max - min) if max != min else 0

def Max_abs_scaler_derivative(x, max):
  return 1 / max if max != 0 else 0

def Robust_scaler_derivative(x, q1, q3):
  return 1 / (q3 - q1) if q3 != q1 else 0

# normalization functions

def Sigmoid_derivative(x):
  return Activation.Sigmoid(x) * (1 - Activation.Sigmoid(x))

def Tanh_derivative(x):
  return 1 - (math.tanh(x))**2

def Binary_step_derivative(x):
  return Sigmoid_derivative(x)

def Softsign_derivative(x):
  return 1 / (1 + abs(x))

def Softmax_derivative(probabilities:list):
  def same(ax):
    return ax * (1 - ax)
  
  def different(ax, aj):
    return -ax * aj
  
  answer = []
  
  for base_index, base_value in enumerate(probabilities):
    for index, value in enumerate(probabilities):
      if base_index == index:
        answer.append(same(value))
      else:
        answer.append(different(base_value, value))
    
  return answer

# rectifier functions

def ReLU_derivative(x):
  return 1 if x > 0 else 0

def Sigmoid_derivative(x):
  return Activation.Sigmoid(x) * (1 - Activation.Sigmoid(x))

def Softplus_derivative(x):
  if x > 10:
    return 1
  elif x < -10:
    return 0
  else:
    return 1 / (1 + math.exp(-x))

def Mish_derivative(x):
  return 1 + Activation.Mish(x) * (1 - Activation.Mish(x))

def Swish_derivative(x):
  return Activation.Swish(x) + Activation.Sigmoid(x) * (1 - Activation.Swish(x))

def Leaky_ReLU_derivative(x, alpha=0.01):
  return 1 if x > 0 else alpha

def ELU_derivative(x, alpha=1.0):
  if x > 0:
    return 1

  elif x < -10:
    return 0

  else:
    return alpha * math.exp(x)

def GELU_derivative(x):
  cdf = 0.5 * (1.0 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
  pdf = math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * x**2) * (1 - cdf**2)
  return cdf + x * pdf

def SELU_derivative(x, alpha=1.67326, scale=1.0507):
  if x > 0:
    return scale

  elif x < -10:
    return 0

  else:
    return scale * (1 if x > 0 else alpha * math.exp(x))

def ReEU_derivative(x):
  if x > 0:
    return 1

  elif x < -10:
    return 0

  else:
    return math.exp(x)

def Linear_derivative(x):
  return 1

def TANDIP_derivative(x):
  if x > 0:
    return 1
  else:
    if x < -20:
      return 0
    else:
      return (math.tanh(x+1) + ( x/math.cosh(x+1)**2 ) + 1)/2
  