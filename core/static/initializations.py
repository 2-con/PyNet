import math
import random

def Glorot_uniform(input_size, output_size):
  final_output_size = 1
  for dim in output_size:
    final_output_size *= dim
  
  output_size = final_output_size
  return random.uniform(-math.sqrt(2 / (input_size + output_size)), math.sqrt(2 / (input_size + output_size)))

def Glorot_normal(input_size, output_size):
  final_output_size = 1
  for dim in output_size:
    final_output_size *= dim
  
  output_size = final_output_size
  return random.normalvariate(0, 2 / (input_size + output_size))

def He_uniform(input_size, output_size):
  final_output_size = 1
  for dim in output_size:
    final_output_size *= dim
  
  output_size = final_output_size
  return random.uniform(-math.sqrt(6 / input_size), math.sqrt(6 / input_size))

def He_normal(input_size, output_size):
  final_output_size = 1
  for dim in output_size:
    final_output_size *= dim
  
  output_size = final_output_size
  return random.normalvariate(0, 2 / input_size)

def Lecun_uniform(input_size, output_size):
  final_output_size = 1
  for dim in output_size:
    final_output_size *= dim
  
  output_size = final_output_size
  return random.uniform(-math.sqrt(3 / input_size), math.sqrt(3 / input_size))

def Lecun_normal(input_size, output_size):
  final_output_size = 1
  for dim in output_size:
    final_output_size *= dim
  
  output_size = final_output_size
  return random.normalvariate(0, 1 / input_size)

def Xavier_uniform_in(input_size, output_size):
  final_output_size = 1
  for dim in output_size:
    final_output_size *= dim
  
  output_size = final_output_size
  return random.uniform(-math.sqrt(6 / input_size), math.sqrt(6 / input_size))

def Xavier_uniform_out(input_size, output_size):
  final_output_size = 1
  for dim in output_size:
    final_output_size *= dim
  
  output_size = final_output_size
  return random.uniform(-math.sqrt(6 / output_size), math.sqrt(6 / output_size))

def Default(input_size, output_size):
  final_output_size = 1
  for dim in output_size:
    final_output_size *= dim
  
  output_size = final_output_size
  return random.uniform(-1, 1)