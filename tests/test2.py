import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax

class deez:
  wa = 1

  def __init__(self,new_wa=None):
    self.wa = new_wa if new_wa is not None else self.wa

print(deez().wa)
print(deez(3).wa)