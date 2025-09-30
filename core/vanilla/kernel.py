import math

class Linear:
  def __init__(self, *args, **kwargs):
    pass
  
  def __call__(self, x1, x2):
    return sum([a*b for a,b in zip(x1, x2)])
  
class Polynomial:
  def __init__(self, constant, degree, *args, **kwargs):
    self.constant = constant
    self.degree = degree
    
  def __call__(self, x1, x2):
    return (sum([a*b for a,b in zip(x1, x2)])+self.constant)**self.degree

class RBF:
  def __init__(self, gamma, *args, **kwargs):
    self.gamma = gamma
    
  def __call__(self, x1, x2):
    magnitude = math.sqrt(sum([(a-b)**2 for a,b in zip(x1, x2)]))
    return math.exp(-self.gamma * magnitude**2)
  
class Sigmoid:
  def __init__(self, constant, gamma, *args, **kwargs):
    self.gamma = gamma
    self.constant = constant
  
  def __call__(self, x1, x2):
    return math.tanh( self.gamma * sum([a*b for a,b in zip(x1, x2)]) + self.constant)
  
class Laplace:
  def __init__(self, gamma, *args, **kwargs):
    self.gamma = gamma
  
  def __call__(self, x1, x2):
    magnitude = math.sqrt(sum([(a-b)**2 for a,b in zip(x1, x2)]))
    return math.exp(-self.gamma * magnitude)