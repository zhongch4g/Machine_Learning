import matplotlib.pyplot as plt
import numpy as np
def func(z):
    """0 or 1
    """
    if z >= 0:
        return 1
    else:
        return 0

data = np.mat([
    [6, -1, 1],
    [-3, 1, 1],
    [0, -1, 0],
    [7, -2, 0]
])
p = data[:, 0:2]
t = data[:, 2]

# %Init
# e = t - a
W = np.mat([0, 0])
b = 0
k = 0
e = np.mat([-1, -1, -1, -1])
e_c = np.mat([0, 0, 0, 0])

while not (e == e_c).all():
    a = func(W * p[k].T + b)
    e[:, k] = t[k] - a
    # %update W and b
    W = W + np.multiply(e[:, k], p[k])
    b = b + e[:, k]
    print k
    print W, b
#     print e, e_c
    if k >= 3:
        k = 0
    else:
        k += 1  

def f(t):
    w = np.array(W)
    return (-(w[0][0] * t) - b)/float(w[0][1])

x = np.array(data[:, 0].T)
y = np.array(data[:, 1].T)
plt.scatter(x[:, :2], y[:, :2], c = 'b',marker = 'o')
plt.scatter(x[:, 2:4], y[:, 2:4], c = 'r',marker = 'o')
plt.axis([-10, 10, -10, 10])
plt.xlabel('X of Set')
plt.ylabel('Y of Set')
plt.title('Single Layer Perceptron')
t1 = np.array(data[:, 0])
plt.legend('10')
plt.plot(t1, f(t1), 'g-')
plt.show()