from tqdm import tqdm
# def readfile(f, t):
#     s = f.read(t)
#     while s[-1]!='\n':
#         s += f.read(1)
#     return s

# f = open('Friendster.txt','r')
# g = open('newFriendster.txt', 'w')
# newid = {}
# index = 0
# for i in tqdm(range(1806067135)):
#     line = f.readline().strip()
#     s = line.split(' ')
#     a = int(s[0])
#     b = int(s[1])
#     if newid.get(a, None)==None:
#         newid[a] = index
#         index += 1
#     if newid.get(b, None)==None:
#         newid[b] = index
#         index += 1
#     g.write(str(newid[a])+' '+str(newid[b])+'\n')
    
# f.close()
# g.close()

# f = open('Orkut.txt','r')
# nodes = [False] * 3072626
# for i in tqdm(range(117185083)):
#     line = f.readline().strip()
#     s = line.split(' ')
#     a = int(s[0])
#     b = int(s[1])
#     if a>=3072626:
#         print(a)
#         break
#     if b>=3072626:
#         print(b)
#         break
#     nodes[a] = True
#     nodes[b] = True
# f.close()
# print(sum(nodes))

f = open('Orkut.txt','r')
g = open('NewOrkut.txt', 'w')
newid = {}
index = 0
for i in tqdm(range(117185083)):
    line = f.readline().strip()
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
    
f.close()
g.close()