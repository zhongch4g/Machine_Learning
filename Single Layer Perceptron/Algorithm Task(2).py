import numpy as np
binary_num = np.mat([
    [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
])
p = binary_num[:, :15]
t = binary_num[:, 15]

def func(z):
    """0 or 1
    """
    if z >= 0:
        return 1
    else:
        return 0

W = np.zeros(15, dtype = 'int8')
b = 0
k = 0
kp = 0
e = np.mat([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
e_c = np.zeros(15, dtype = 'int')
while not (e == e_c).all():
    a = func(W * p[kp].T + b)
    e[:, k] = t[kp] - a
    # %update W and b
    W = W + np.multiply(e[:, k], p[kp])
    b = b + e[:, k]
    print k
    print W, b
#     print e, e_c
    if k >= 14:
        k = 0
    else:
        k += 1 
    if kp >= 9:
        kp = 0
    else:
        kp += 1 

def test(x):
    k = func(x[0] * -2 + x[1] * 0 + x[2] * -1 + x[3] * 1 + x[4] * -2 + x[5] * -1 + x[6] * 3 + x[7] * -1 + x[8] * -1 + x[9] * 6 + x[10] * 1 + x[11] * 0 + x[12] * -1 + x[13] * 0 + x[14] * 0)
    if k == 1:
        print "this number is even"
    else:
        print "this number is odd"
testc = map(test, np.array(p))