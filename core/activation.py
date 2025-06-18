import math

# normalization functions
def Sigmoid(x):
  # if x < -50.0:
  #   return 0.0
  # elif x > 50.0:
  #   return 1.0
  return 1 / (1 + math.exp(-x))

def Tanh(x):
  return math.tanh(x)

def Binary_step(x):
  return 1 if x > 0 else 0

def Softsign(x):
  return x / (1 + abs(x))

def Softmax(x):
  exp_x = [math.exp(i) for i in x]
  return [i / sum(exp_x) for i in exp_x]

# rectifiers

def ReLU(x):
  return max(0,x)

def Softplus(x):
  if x>10:
    return x
  elif x<-10:
    return 0
  return math.log(1 + math.exp(x))

def Mish(x):
  if x>10:
    return x
  elif x<-10:
    return 0
  return x * math.tanh(math.log(1 + math.exp(x)))

def Swish(x):
  return x * Sigmoid(x)

def Leaky_ReLU(x, alpha=0.01):
  return max(alpha * x, x)

def ELU(x, alpha=1.0):
  if x > 0:
    return x
  else:
    if x < -10:
      return -1
    return alpha * (math.exp(x) - 1)

def GELU(x):
  if x>10:
    return x
  elif x<-10:
    return 0
  return (x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))))/2

def SELU(x, alpha=1.67326, scale=1.0507):
  return scale * (x if x > 0 else alpha * (math.exp(x) - 1))

def Linear(x):
  return x

def ReEU(x):
  if x>10:
    return x
  elif x<-10:
    return 0
  return min(math.exp(x),max(1,x+1))

def TANDIP(x):
  return x * (math.tanh(x+1)+1)/2
