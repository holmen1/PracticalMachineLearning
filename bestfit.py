from statistics import mean
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def create_dataset(N,variance,step=2,correlation=False):
    val = 1
    ys = []
    for i in range(N):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step

    xs = [i for i in range(len(ys))]
    
    return np.array(xs, dtype=np.float64),np.array(ys,dtype=np.float64)


def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]

    squared_error_regr = sum((ys_line - ys_orig) * (ys_line - ys_orig))
    squared_error_y_mean = sum((y_mean_line - ys_orig) * (y_mean_line - ys_orig))

    r_squared = 1 - (squared_error_regr/squared_error_y_mean)

    return r_squared



xs, ys = create_dataset(100,22,2,correlation='neg')
#xs, ys = create_dataset(100,22,2)
X = np.vstack([xs, np.ones(len(xs))]).T

m, c = np.linalg.lstsq(X, ys)[0]
print(m, c)

# or numpy linalg
XX = np.matmul(X.T,X)
Xy = np.matmul(X.T,ys)
XXinv = np.linalg.inv(XX)
print(np.matmul(XXinv,Xy))

regression_line = [(m*x)+c for x in xs]
r_squared = coefficient_of_determination(ys,regression_line)
print(r_squared)

plt.scatter(xs,ys,color='#003F72', label = 'data')
plt.plot(xs, regression_line, label = 'regression line')
plt.legend(loc=4)
plt.show()
