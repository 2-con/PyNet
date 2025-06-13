"""
Visualizer
-----
Visualizes 2D/1D arrays through the terminal.
"""

import time

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

def numerical_display(input, **kwargs):
  """
  Numerical Display
  -----
    Displays the 1D/2D array nicely
  -----
  Args
  -----
  - (Optional) title    (string) : the title of the array
  - (Optional) decimals (string) : displays how much decimals to show
  
  Returns
  -----
    None
  """
  
  title = kwargs.get('title', '')
  decimals = kwargs.get('decimals', 100)
  
  maxnum = 0
  pad = 0

  # get the longest item
  for row in input:
    try:
      for item in row:
        maxnum = len(str(item)) if len(str(item)) > maxnum else maxnum
    except:
      maxnum = len(str(row)) if len(str(row)) > maxnum else maxnum

  pad = maxnum

  if type(input[0]) != list:
    print("[ ", end='')
    for row in input:

      print(round(row,decimals), end=' ' + ' ' * (pad-len(str(row))))
    print("]", end='')

  else:
    print("[ " + title)
    for row in input:
      print("  [ ", end='')
      for item in row:
        try:
          print(round(item,decimals), end=' ' + ' ' * (pad-len(str(item))))
        except:
          print(item, end=' ' + ' ' * (pad-len(str(item))))
      print("]", end='')
      print()
    print("]")
  
  print()

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