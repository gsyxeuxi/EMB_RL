import numpy as np
import matplotlib.pyplot as plt

import time
import csv

x = np.array([-10, -500, -10, -500, 0, 1e-5, 0], dtype=np.float64)
print(np.concatenate((x[:2], x[-3:])))