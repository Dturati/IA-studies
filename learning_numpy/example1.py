p = print

import numpy as np

a = np.arange(16).reshape(4, 4)
p(a)
p(a.shape)
p(a.size)

p(np.atleast_2d(a))