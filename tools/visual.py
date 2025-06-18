"""
Visualizer
-----
Visualizes 2D/1D arrays through the terminal.
"""

import time
from tools.arraytools import shape
from tools.math import sgn

def image_display(input, **kwargs):
  """
  Black and White Display
  -----
    Displays the 2D/1D array as a monochromatic image
  -----
  Args
  -----
  - (Optional) boundary        (float) : the boundary seperating light from dark values
  - (Optional) upper_tolerance (float) : pixels with values higher than the boundary + upper_tolerance will be displayed as "■"
  - (Optional) lower_tolerance (float) : pixels with values lower than the boundary - lower_tolerance will be displayed as "□"
  - (Optional) title           (string) : the title of the image
  
  Returns
  -----
    None
  """
  upper_tolerance = kwargs.get('upper_tolerance', 0)
  lower_tolerance = kwargs.get('lower_tolerance', 0)
  boundary        = kwargs.get('boundary', 0)
  title = kwargs.get('title', '')
  
  print(title)
  
  if type(input[0]) == list: # probably a 2D array
    for row in input:
      for pixel in row:
        if pixel > boundary + upper_tolerance:
          print("■", end=" ")

        elif pixel < boundary - lower_tolerance:
          print("□", end=" ")

        else:
          print("•", end=" ")
      print()

  else: # probably a 1D array
    for pixel in input:
      if pixel > boundary + upper_tolerance:
        print("■", end=" ")

      elif pixel < boundary - lower_tolerance:
        print("□", end=" ")

      else:
        print("•", end=" ")
    print()

def numerical_display(data, pad=2, start_indent=0, decimals=10, **kwargs):
  """
  Numerical Display
  -----
    Displays the 1D/2D array nicely
  -----
  Args
  -----
  - data                    (list or tuple) : the data to be shown
  - (optional) pad          (int)           : the number of spaces to indent the data
  - (optional) start_indent (int)           : the initial number of spaces to indent the data
  - (Optional) decimals     (string)        : displays how much decimals to show, assumeing its a float
  
  Returns
  -----
    None
  """
  data_shape = shape(data)
  indent = ' ' * start_indent
  # do not pass these as kwargs except during recursion
  maxnum = kwargs.get('maxnum', 0)
  padding = kwargs.get('padding', [])
  depth = kwargs.get('depth', 0)
  
  # padding
  # multidimensional
  for img in data:
    if type(img) in (list, tuple) and len(shape(img)) == 2:
      
      maxnum = 0
      
      for point in img:
        for subpoint in point:
          
          if type(subpoint) in (float, int):
            
            maxnum = len(str(round(subpoint, decimals))) if len(str(round(subpoint, decimals))) > maxnum else maxnum
        
          else:
            maxnum = len(str(subpoint)) if len(str(subpoint)) > maxnum else maxnum
        
      padding.append(maxnum)
  
  #2d
  if type(data) in (list, tuple) and len(shape(data)) == 2 and depth==0:
      
    maxnum = 0
    
    for point in data:
      for subpoint in point:
        
        if type(subpoint) in (float, int):
          
          maxnum = len(str(round(subpoint, decimals))) if len(str(round(subpoint, decimals))) > maxnum else maxnum
      
        else:
          maxnum = len(str(subpoint)) if len(str(subpoint)) > maxnum else maxnum
      
    padding.append(maxnum)
  
  # 1d
  if type(data) in (list, tuple) and len(shape(data)) == 1 and depth==0:
    maxnum = 0
    padding = []
    for point in data:
        
      if type(point) in (float, int):
        
        maxnum = len(str(round(point, decimals))) if len(str(round(point, decimals))) > maxnum else maxnum
    
      else:
        maxnum = len(str(point)) if len(str(point)) > maxnum else maxnum
      
    padding.append(maxnum)
    
  # display
  if all(type(item) not in (list, tuple) for item in data):
    
    print(indent + "[ ", end='')
    maxnum = max(padding)
    
    for index, point in enumerate(data):
      
      if type(point) in (float, int):
        print(round(point, decimals), end=' ' + ' ' * ( maxnum - len(str(round(point, decimals)))) )
        
      elif type(point) == str:
        print(f"'{point}'", end=' ' + ' ' * ( maxnum - len(str(point))) )
        
      elif type(point) == bool:
        print(point, end=' ' + ' ' * ( maxnum - len(str(point))) )
        
      else:
        print(f"<{point.__name__}>", end=' ' + ' ' * ( maxnum - len(str(point))) )
        
    print("]")
    
  else:
    print(indent + '[')
    for i in range(data_shape[-1]):
      numerical_display(data[i], pad, start_indent=start_indent + pad, decimals=decimals, maxnum=maxnum, padding=padding, depth=depth + 1)
      
    print(indent + ']')

def char_display(string, speed, wrap = 50):
  """
  Charachter Display
  -----
    Displays the entire string charachter by charachter
  -----
  Args
  -----
  - string (string) : the string to display
  - speed  (float)  : how fast to print
  - wrap   (int)    : how far the text can go before wrapping
  
  Returns
  -----
    None
  """
  
  count = 0
  for i in string:
    count += 1
    time.sleep(speed)
    
    if count % wrap == 0:
      print()
    
    print(i, end='', flush=True)
  print()

def word_display(string, speed, wrap = 50):
  """
  Charachter Display
  -----
    Displays the entire string word by word
  -----
  Args
  -----
  - string (string) : the string to display
  - speed  (float)  : how fast to print
  - wrap   (int)    : how far the text can go before wrapping
  
  Returns
  -----
    None
  """
  count = 0
  for i in string.split():
    count += 1
    time.sleep(speed)
    
    if count % wrap == 0:
      print()
    
    print(i, end=' ', flush=True)
  print()