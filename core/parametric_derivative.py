import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from system.defaults import *
import math
import core.derivative as Derivative

def ELU_parametric_derivative(x, variable, alpha=ELU_alpha_default, **kwargs):
  if x > 0:
    return 0
  else:
    return math.exp(x) - 1
  
def SELU_parametric_derivative(x, variable, alpha=SELU_alpha_default, beta=SELU_beta_default, **kwargs):
  if variable == 'alpha':
    if x > 0:
      return 0
    else:
      return alpha * (math.exp(x) - 1)
  elif variable == 'beta':
    if x > 0:
      return 0
    else:
      return beta * (math.exp(x) - 1)
  
def SiLU_parametric_derivative(x, variable, alpha=SiLU_alpha_default, **kwargs):
  return x**2 * Derivative.Sigmoid_derivative(alpha * x)

def PReLU_parametric_derivative(x, variable, alpha=PReLU_alpha_default, **kwargs):
  if x > 0:
    return 0
  else:
    return x

