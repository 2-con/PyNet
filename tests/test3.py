import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

a = [1,2,3,4]
for thing in a:
  thing -= 1
print(a)