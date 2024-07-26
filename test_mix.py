import tensorflow as tf
import numpy as np
import math

import numpy as np
import tensorflow as tf

filename = f"fi_info_10_iter_300.npz"


with np.load(filename) as data:
    C = data['fi_info_10']
print(C)

print(np.diag([4,4,4,4]))
