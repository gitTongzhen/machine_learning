import matplotlib
# matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
x = [6,8,10,14,18]
y = [7,9,13,17.5,18]


# plt.plot(x, y, 'k.')
# plt.show()

from sklearn import linear_model        #表示，可以调用sklearn中的linear_model模块进行线性回归。
import numpy as np
model = linear_model.LinearRegression()
X = np.array(x)[:, np.newaxis]
Y = np.array(y)

print(X)
print(Y)
model.fit(X, Y)

print(model.coef_)
print(model.predict([[12]]))
