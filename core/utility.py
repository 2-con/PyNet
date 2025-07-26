"""
Utility
=====
  Utility functions that might be useful to have, functions here typically relate to the internal workings
  of the APIs, or are in some way critical in how the APIs work. Functions with a more spesific field of application
  are stored in their respective modules.
"""

def split(data, second_size):
  """
  Split
  -----
    Splits a list into two lists of configurable lengths, 
    used for splitting data into training and testing sets
  -----
  Args
  -----
  - second_size (float) : controls how much of the original list will go towards the second list
  -----
  Returns
  -----
    (1st half, second half)
  """
  first_size = int((1 - second_size) * len(data))
  return data[:first_size], data[first_size:]

def do_nothing(*args, **kwargs) -> 0:
  """
  Do Nothing
  -----
    Does nothing and returns 0
  -----
  Args
  -----
  - args (positional) : positional arguments
  - kwargs (keyword)  : keyword arguments
  -----
  Returns
  -----
    0
  """
  return 0
