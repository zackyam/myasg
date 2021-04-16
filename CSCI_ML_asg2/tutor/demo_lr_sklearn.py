from numpy import *
import matplotlib.pyplot as plt
import seaborn

seaborn.set()
from sklearn import linear_model
from tu3utils import *


points = genfromtxt('imports-85.data', delimiter=',', usecols=(21, 25))
points = points[~isnan(points).any(axis=1)]
points[:, 1] = points[:, 1] / 1000
points[:, 0] = normalize_feature(points[:, 0])
x = points[:, 0]

x_to_train = x[:, newaxis]
y_to_train = points[:, 1]


clf = linear_model.SGDRegressor(loss='squared_loss')

clf.fit(x_to_train, y_to_train)
y_fit = clf.predict(x_to_train)
plt.plot(x_to_train.squeeze(), y_to_train, 'o')
plt.plot(x_to_train.squeeze(), y_fit)
plt.xlabel(r'Normalized Horse Power', fontsize=20)
plt.ylabel(r'Price (K\$)', fontsize=20)
plt.grid(True)
ax = plt.gca()
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ymajor_ticks = arange(0, 51, 5)
xmajor_ticks = arange(-2, 7, 2)

ax.set_xticks(xmajor_ticks)
ax.set_yticks(ymajor_ticks)

ax.grid(True)

plt.savefig('sklearn', bbox_inches='tight', pad_inches=0)
plt.show()

