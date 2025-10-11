import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.lab.activations import Sigmoid

print(Sigmoid.forward(3))
