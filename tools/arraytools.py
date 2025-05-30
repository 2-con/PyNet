"""
Array Tools
-----
A collection of tools for working with arrays, mainly 1D and 2D arrays
"""
  
import random

def convolve2d(matrix, kernel):
    """
    Convolve a 2D matrix with a 2D kernel.
    
    Args:
      matrix (list of list of int): The input matrix.
      kernel (list of list of int): The kernel to convolve with.
    
    Returns:
      list of list of int: The convolved matrix.
    """
    m_rows = len(matrix)
    m_cols = len(matrix[0])
    k_rows = len(kernel)
    k_cols = len(kernel[0])
    
    # Calculate the dimensions of the output matrix
    output_rows = m_rows - k_rows + 1
    output_cols = m_cols - k_cols + 1
    
    # Initialize the output matrix with zeros
    output = [[0] * output_cols for _ in range(output_rows)]
    
    # Perform convolution
    for i in range(output_rows):
      for j in range(output_cols):
        for ki in range(k_rows):
          for kj in range(k_cols):
            output[i][j] += matrix[i + ki][j + kj] * kernel[ki][kj]
    
    return output

def reshape(input, width, height):
  """
  Reshape
  -----
    Reshapes a 2D array and returns a 2D array of specifed dimensions
    
    negative values means that the shape will be counted backwards simmilar to
    string indexing / slicing
  -----
  Args
  -----
  height (int)  : the height of the output
  width  (int)  : the width of the output
  
  Returns
  -----
    2D Array
  """
  # pool everything
  pile = []
  for row in input:
    try:
      for item in row:
        pile.append(item)
    except:
      pile.append(row)

  # infer the shape
  if height < 0:
    height = (len(pile) // abs(width)) + (height+1)

  if width < 0:
    width = (len(pile) // abs(height)) + (width+1)

  # reshape the pile
  answer = []
  for a in range(height):
    row = []
    for b in range(width):
      try:
        row.append(pile[0])
        pile.pop(0)
      except:
        raise ValueError(f"Cannot reshape matrix into a {width}x{height} matrix")

    answer.append(row[:])

  # check if it can be converted into anything other than a redundant 2D array
  if len(answer) == 1:
    return answer[0]

  if len(answer) == 1 and len(answer[0]) == 1:
    return answer[0][0]

  return answer

def flatten(input):
  """
  Flatten
  -----
    Flattens any 2D array into a 1D array
  -----
  Args
  -----
  input (2D array) : the array to flatten
  
  Returns
  -----
    1D List
  """
  answer = []
  if type(input[0]) == list:
    for layer in input:
      for element in layer:
        answer.append(element)

  else:
    raise TypeError("input must be a 2D array")

  return answer

def transpose(input):
  """
  Transpose
  -----
    Transposes a 2D array (rows become columns, columns become rows)
  -----
  Args
  -----
  input (2D array) : the array to transpose
  
  Returns
  -----
    2D array
  """
  answer = []
  for i in range(len(input[0])):
    row = []
    for j in range(len(input)):
      row.append(input[j][i])
    answer.append(row)

  return answer

def array_equal(input1, input2):
  """
  Array equality
  -----
    Compares if two 2D arrays are the same dimentions
  -----
  Args
  -----
  input1 (2D array) : the first array
  input2 (2D array) : the second array
  
  Returns
  -----
    Bool
  """
  if len(input1) != len(input2):
    return False

  for i in range(len(input1)):
    if len(input1[i]) != len(input2[i]):
      return False

    for j in range(len(input1[i])):
      if input1[i][j] != input2[i][j]:
        return False

  return True

def generate_array(*args, **kwargs):
  """
  Generate array
  -----
    generate an array of specified dimensions
  -----
  Args
  -----
  args (int) : the dimensions of the array
  value (int) : the default value of the array
  
  Returns
  -----
    List / Array
  """

  value = kwargs.get('value', 0)

  answer = [value for _ in range(args[0])]

  for dimention in args[1:]:
    if dimention < 0:
      raise ValueError("Dimensions cannot be negative")

    answer = [answer[:] for _ in range(dimention)]

  return answer

def generate_random_array(*args, **kwargs):
  """
  Genrate random array
  -----
    Creates an array of specified dimensions with random values
  -----
  Args
  -----
  args (int) : the dimensions of the array
  min  (int) : the minimum value of the array
  max  (int) : the maximum value of the array
  
  Returns
  -----
    List / Array
  """

  min = kwargs.get('min', 0)
  max = kwargs.get('max', 1)

  if not all(isinstance(dim, int) for dim in args):
    raise TypeError("All dimensions must be integers.")
  if not all(dim > 0 for dim in args):
    raise ValueError("All dimensions must be greater than 0.")
  if min > max:
    raise ValueError("min_val must be less than or equal to max_val.")

  def _generate_inner(dimensions):
    """
    Recursive helper function to generate the nested lists.
    """
    if not dimensions:
      return random.uniform(min, max)  # Base case: 0 dimensions
    else:
      return [_generate_inner(dimensions[1:]) for _ in range(dimensions[0])]

  return _generate_inner(args)

def shape(input):
  """
  Shape
  -----
    Returns the shape of an array, caps at 2D arrays
  -----
  Args
  -----
  input (2D array) : the array to count
  
  Returns
  -----
  (X size, Y size)
  """
  rows = 0
  pixel = 0

  if input == None:
    return 0,0

  if type(input) not in (list, tuple):
    return 1,1

  if type(input[0]) not in (list, tuple):
    return len(input), 1

  return len(input[0]), len(input)

def flip(input, axis = 'X'):
  """
  Flip
  -----
    Flips a 2D array along the X or Y axis
  -----
  Args
  -----
  input (2D array) : the array to flip
  axis (string) : the axis to flip the array along ('X' or 'Y')
  
  Returns
  -----
    2D array
  """
  if axis == 'X':
    return [row[::-1] for row in input]

  elif axis == 'Y':
    return input[::-1]

  else:
    raise ValueError("axis must be 'X' or 'Y'")

def size(input):
  """
  Size
  -----
    Returns the quantity of elements in a 2D array
  -----
  Args
  -----
  input (2D array) : the array to count
  
  Returns
  -----
    Integer
  """
  return len(flatten(input))

def clear(input, **kwargs):
  """
  Clear
  -----
    Clears a 1D or a 2D array
  -----
  Args
  -----
  input (1D or 2D array) : the array to clear
  
  Returns
  -----
    1D / 2D array
  """
  
  value = kwargs.get('value', 0)
  
  for i in range(len(input)):
    
    try:
      for j in range(len(input[i])):
        input[i][j] = value
    except:
      input[i] = value

  return input

def mirror(input, axis:str):
  """
  Mirror
  -----
    Mirrors a 2D array along the X or Y axis
  -----
  Args
  -----
  input (2D array) : the array to mirror
  axis (string) : the axis to mirror the array along 'X' or 'Y'
  
  Returns
  -----
    2D array
  """
  if axis == 'Y':
    return [row[::-1] for row in input]

  elif axis == 'X':
    return input[::-1]

  else:
    raise ValueError("axis must be 'X' or 'Y'")