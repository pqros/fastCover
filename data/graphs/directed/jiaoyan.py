'''
产生一个文件的md5码
'''
import os
import hashlib
import sys
filename = sys.argv[1]
def md5sum(file):
    m=hashlib.md5()
    if os.path.isfile(file):
        f=open(file,'rb')
        for line in f:
            m.update(line)
        f.close
    else:
        m.update(file)
    return (m.hexdigest())

print(md5sum(filename))