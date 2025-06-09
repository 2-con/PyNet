import sys
import os

# Get the directory of the current script (test.py)
current_script_dir = os.path.dirname(__file__)

# Navigate up one level to the 'PyNet' directory
# If test.py is in PyNet/tests/, then '..' takes us to PyNet/
pynet_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))

# Add the PyNet root directory to Python's module search path
sys.path.append(pynet_root_dir)

from tools import arraytools
from tools.visual import numerical_display

twodim = [
  [1,2,3],
  [4,5,6],
  [7,8,9]
  ]

bar = [
  [1,2,3],
  [4,5,6],
  ]

biggertwodim = [
  [ 1, 2, 3, 4],
  [ 5, 6, 7, 8],
  [ 9,10,11,12],
  [13,14,15,16]
  ]

threedim = [
  [
    [ 1, 2, 3, 4],
    [ 5, 6, 7, 8],
    [ 9,10,11,12],
    [13,14,15,16]
  ],
  [
    [ 1, 2, 3, 4],
    [ 5, 6, 7, 8],
    [ 9,10,11,12],
    [13,14,15,16]
  ]
]

print(f""" 
    None  {arraytools.shape(None)}
       1  {arraytools.shape(1)}
 [1,2,3]  {arraytools.shape([1,2,3])}
  twodim  {arraytools.shape(twodim)}
     bar  {arraytools.shape(bar)}
threedim  {arraytools.shape(threedim)}
""")