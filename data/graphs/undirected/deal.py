# f = open('Youtube.txt', 'r')
# g = open('newYoutube.txt' ,'w')
# while True:
#     line = f.readline()
#     if not line:
#         break
#     g.write(line)
#     f.readline()
# f.close()
# g.close()

# f = open('newdblp.txt', 'r')
# g = open('newnewdblp.txt' ,'w')
# while True:
#     line = f.readline()
#     if not line:
#         break
#     g.write(line.replace(' ', '\t'))
# f.close()
# g.close()

# f = open('Nethept.txt', 'r')
# g = open('newNethept.txt' ,'w')
# edges = {}
# while True:
#     line = f.readline()
#     if not line:
#         break
#     line = line.strip()
#     s = line.split(' ')
#     a = int(s[0])
#     b = int(s[1])
#     if a not in edges.keys():
#         if b not in edges.keys():
#             edges[a] = set()
#             edges[a].add(b)
#             g.write(line+'\n')
#         else:
#             if a in edges[b]:
#                 continue
#             else:
#                 edges[b].add(a)
#                 g.write(line+'\n')
#     else:
#         if b in edges[a]:
#             continue
#         else:
#             edges[a].add(b)
#             g.write(line+'\n')
# f.close()
# g.close()

# f = open('/home/wgh/IM-src/data/friendster/com-friendster.ungraph.txt', 'r')
# g = open('Friendster.txt' ,'w')
# f.readline()
# f.readline()
# f.readline()
# f.readline()
# t = 1024 * 1024
# i = 0
# while True:
#     i += 1
#     if i%100==0:
#         print(i)
#     line = f.read(t)
#     if not line:
#         break
#     g.write(line.replace('\t', ' '))
# f.close()
# g.close()

# from tqdm import tqdm
# f = open('Orkut.txt', 'r')
# g = open('newOrkut.txt' ,'w')
# for i in tqdm(range(117185083)):
#     line = f.readline()
#     s = line.strip().split(' ')
#     a = int(s[0])
#     b = int(s[1])
#     g.write(str(a-1)+' '+str(b-1)+'\n')
# f.close()
# g.close()

# import os

# def readfile(f, t):
#     line = f.read(t)
#     while line[-1]!='\n':
#         line += f.read(1)
#     return line

# t = 1024 * 1024
# for dirnames, dirs, files in os.walk('.'):
#     for file in files:
#         if file.endswith('.txt'):
#             if file=='Friendster.txt':
#                 continue
#             print(file)
#             with open(file, 'r') as f:
#                 nodes = set()
#                 max_id = 0
#                 while True:
#                     line = f.readline()
#                     if not line:
#                         break
#                     line = line.replace('\n', ' ')
#                     s = line.split(' ')
#                     for c in s:
#                         if c:
#                             a = int(c)
#                             nodes.add(a)
#                             max_id = max(max_id, a)
#             print(max_id, len(nodes))

