'''
统计一个文件的行数
'''
import sys
filename = sys.argv[1]
with open(filename,'r') as f:
    #     count = 0
    #     while(True):
    #         if count%10000==0:
    #             print(int(count/10000), end=' ')
    #         f.readline()
    #         count += 1
        count = 0
        while True:
            buffer = f.read(1024*8192)
            if not buffer:
                break
            count += buffer.count('\n')
print(count)

