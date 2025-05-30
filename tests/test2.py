import sys
import os

# Get the directory of the current script (test.py)
current_script_dir = os.path.dirname(__file__)

# Navigate up one level to the 'PyNet' directory
# If test.py is in PyNet/tests/, then '..' takes us to PyNet/
pynet_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))

# Add the PyNet root directory to Python's module search path
sys.path.append(pynet_root_dir)

from tools.utility import timer

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

test1 = [
  [1,1,1],
  [1,1]
]

test2 = [
  [1,1,1],
  [1,1,1],
  [1,1,1]
]

print(convolve2d(test2, test1))

"""
def apply(self, input):
    answer = []
    
    m_rows = len(input)
    m_cols = len(input[0])
    k_rows = len(self.kernel)
    k_cols = len(self.kernel[0])
    
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
            output[i][j] += input[i + ki][j + kj] * input[ki][kj]
    
    return output

"""