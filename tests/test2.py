import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import jax

ts = time.perf_counter()

a = jax.lax.fori_loop(jax.numpy.int64(0), jax.numpy.int64(int(1E9)), lambda _, i: i + 1, 0)

# a = 0
# for i in range(int(1E7)):
#   a+=1

te = time.perf_counter()
print(te-ts)
print(a)
