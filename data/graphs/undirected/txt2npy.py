import numpy as np
import time
name = 'Friendster'
s = time.time()
l = np.loadtxt(name+'.txt',delimiter=' ')
e = time.time()
print(e-s)

s = time.time()
ll = np.array(l)
np.save(name+'.npy', ll)
e = time.time()
print(e-s)

# s = time.time()
# ll = np.load(name+'.npy')
# e = time.time()
# print(e-s)
# print(ll.shape)