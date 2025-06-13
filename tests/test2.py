import sys
import os

current_script_dir = os.path.dirname(__file__)
pynet_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(pynet_root_dir)

from core.utility import split

rizz = [1,2,3,4,5,6,7,8,9,10]

skibidi, toilet = split(rizz, 0.8)

print(skibidi)
print(toilet)
# print(rizz[:1])
# print(rizz[1:])
