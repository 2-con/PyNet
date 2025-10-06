import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

a = "test"

for i in range(3):
  a += "\b"
  
a += "          "
print(a)