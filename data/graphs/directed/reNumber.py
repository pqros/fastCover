import os

for rt, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('txt'):
            print(file)
            f = open(rt+'/'+file,'r')
            g = open(file[0:-4]+'.new.txt', 'w')
            newid = {}
            index = 0
            for line in f.readlines():
                line = line.strip()
                s = line.split(' ')
                a = int(s[0])
                b = int(s[1])
                if newid.get(a, None)==None:
                    newid[a] = index
                    index += 1
                if newid.get(b, None)==None:
                    newid[b] = index
                    index += 1
                g.write(str(newid[a])+' '+str(newid[b])+'\n')
            print(max(newid.keys()))
            f.close()
            g.close()