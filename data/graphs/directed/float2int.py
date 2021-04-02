import numpy as np
import os
import time
for rt, dirs, files in os.walk('.'):
    for file in files:
        if '.npy' in file:
            name = file[0:-4]
            print(file)
            s = time.time()
            l = np.load(name+'.npy')
            e = time.time()
            print(e-s)

            s = time.time()
            ll = l.astype(np.int32)
            np.save('int/'+name+'.npy', ll)
            e = time.time()
            print(e-s)
