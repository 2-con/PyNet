"""
Logic
-----
Contains logic functions that are not built-in python functions.
"""

def xor(a, b):
  return a != b

def nor(a, b):
  return not (a or b)

def nand(a, b):
  return not (a and b)

def xnor(a, b):
  return not xor(a, b)

def nimp(a, b):
  return a and not b

def imp(a, b):
  return not a or b

def majority(*args):
  count = 0
  for i in args:
    if i:
      count += 1
      
  return count > len(args) / 2
