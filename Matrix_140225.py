import numpy as np

n = 10

m = np.diag(np.arange(n-1, 0, -1), k=1)
m = np.flip(m, axis=-1)
m = np.rot90(m, k=-2)

print(m)