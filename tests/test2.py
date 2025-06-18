import sys
import os
current_script_dir = os.path.dirname(__file__)
pynet_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(pynet_root_dir)

from tools.arraytools import generate_random_array as gen
from tools.visual import numerical_display as prnt

prnt(gen(2,2,1,1,1))