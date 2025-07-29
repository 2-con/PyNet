import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
from system.defaults import *

# normalization functions
def Sigmoid(x, **kwargs):
  # if x < -50.0:
  #   return 0.0
  # elif x > 50.0:
  #   return 1.0
  return 1 / (1 + math.exp(-x))

def Tanh(x, **kwargs):
  return math.tanh(x)

def Binary_step(x, **kwargs):
  return 1 if x > 0 else 0

def Softsign(x, **kwargs):
  return x / (1 + abs(x))

def Softmax(x, **kwargs):
  exp_x = [math.exp(i) for i in x]
  return [i / sum(exp_x) for i in exp_x]

# rectifiers

def ReLU(x, **kwargs):
  return max(0,x)

def Softplus(x, **kwargs):
  if x>10:
    return x
  elif x<-10:
    return 0
  return math.log(1 + math.exp(x))

def Mish(x, **kwargs):
  if x>10:
    return x
  elif x<-10:
    return 0
  return x * math.tanh(math.log(1 + math.exp(x)))

def Swish(x, **kwargs):
  return x * Sigmoid(x)

def Leaky_ReLU(x, **kwargs):
  return max(0.1 * x, x)

def GELU(x, **kwargs):
  if x>10:
    return x
  elif x<-10:
    return 0
  return (x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))))/2

def Linear(x, **kwargs):
  return x

def ReEU(x, **kwargs):
  if x>10:
    return x
  elif x<-10:
    return 0
  return min(math.exp(x),max(1,x+1))

def TANDIP(x, **kwargs):
  return x * (math.tanh(x+1)+1)/2

# parametric functions

def ELU(x, alpha=ELU_alpha_default, **kwargs):
  if x > 0:
    return x
  else:
    if x < -10:
      return -1
    return alpha * (math.exp(x) - 1)
  
def SELU(x, alpha=SELU_alpha_default, beta=SELU_beta_default, **kwargs):
  return beta * (x if x > 0 else alpha * (math.exp(x) - 1))

def PReLU(x, alpha=PReLU_alpha_default, **kwargs):
  return max(alpha * x, x)

def SiLU(x, alpha=SiLU_alpha_default, **kwargs):
  return x * Sigmoid(alpha * x)

