"""
Math
-----
Contains useful math functions that are not part of Numpy or Python
"""

import math

def sgn(x):
  """Returns the sign of a number."""
  if x > 0:
    return 1
  elif x < 0:
    return -1
  else:
    return 0

def clamp(x, a, b):
  """Clamps a value within a specified range."""
  return max(min(x, b), a)

def lerp(a, b, t):
  """Linear interpolation between two values."""
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

