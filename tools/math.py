"""
Math
-----
Contains useful math functions that are not part of Numpy or Python
"""

import math

def sgn(x):
  """
  Sign
  -----
    Returns the sign of a number. it eitehr returns 1 if the number is positive, -1 if the number is negative, or 0 if the number is 0.
  -----
  Args:
    x (int / float): The number to check the sign of.
  -----
  Returns:
    int: The sign of the number.
  """
  if x > 0:
    return 1
  elif x < 0:
    return -1
  else:
    return 0

def clamp(x, a, b):
  """
  Clamp
  -----
    Clamps a value within a specified range.
  -----
  Args:
    a (int / float): The minimum value of the range.
    b (int / float): The maximum value of the range.
    x (int / float): The value to clamp.
  """
  return max(min(x, b), a)

def lerp(a, b, t):
  """
  Linear Interpolation
  -----
    Linear interpolation between two values.
  -----
  Args:
    a (int / float): The start value.
    b (int / float): The end value.
    t (float): The interpolation factor, typically between 0 and 1.  
  """
  return a + t * (b - a)

def smoothstep(edge0, edge1, x):
  """Smooth Hermite interpolation between 0 and 1."""
  if x <= edge0:
    return 0
  if x >= edge1:
    return 1
  t = (x - edge0) / (edge1 - edge0)
  return t * t * (3 - 2 * t)

def degrees_to_radians(degrees):
  """Converts degrees to radians."""
  return degrees * math.pi / 180

def radians_to_degrees(radians):
  """Converts radians to degrees."""
  return radians * 180 / math.pi

def distance_2d(x1, y1, x2, y2):
  """Calculates the 2D Euclidean distance between two points."""
  return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def distance_3d(x1, y1, z1, x2, y2, z2):
  """Calculates the 3D Euclidean distance between two points."""
  return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def manhattan_distance(x1, y1, x2, y2):
  """Calculates the Manhattan distance between two points."""
  return abs(x2 - x1) + abs(y2 - y1)

def manhattan_distance_3d(x1, y1, z1, x2, y2, z2):
  """Calculates the 3D Manhattan distance between two points."""
  return abs(x2 - x1) + abs(y2 - y1) + abs(z2 - z1)

def ackermann(x,y):
  """
  Ackermann
  -----
    Computes the Ackermann function. this is not a practical tool but rather, an easter egg.
    be aware that this function notorious for its hyper exponential growth.
  -----
  Args:
    x (int): The first parameter of the Ackermann function.
    y (int): The second parameter of the Ackermann function.
  -----
  Returns:
    int: The result of the Ackermann function.
  """
  if x <= 0:
    return y + 1
  elif y <= 0:
    return ackermann(x-1, 1)
  else:
    return ackermann(x-1, ackermann(x, y-1))

def factorial(x):
  """
  Factorial
  -----
    Computes the factorial of a number.
  -----
  Args:
    x (int): The number to compute the factorial of.
  -----
  Returns:
    int: The factorial of the number.
  """
  if x == 0:
    return 1
  else:
    return x * factorial(x-1)
