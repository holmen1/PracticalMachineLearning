from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)

xs = [1,2,3,4,5]
ys = [5,4,6,5,6]

#plt.scatter(xs,ys)
#plt.show()


xs = np.array([1,2,3,4,5], dtype=np.float64)
ys = np.array([5,4,6,5,6], dtype=np.float64)
X = np.vstack([xs, np.ones(len(xs))]).T
#X = np.concatenate((xs, np.ones(np.shape(xs))), axis=1)
print(X)

m, c = np.linalg.lstsq(X, ys)[0]
print(m, c)

plt.plot(xs, ys, 'o', label='Original data', markersize=10)
plt.plot(xs, m*xs + c, 'r', label='Fitted line')
plt.legend()
plt.show()