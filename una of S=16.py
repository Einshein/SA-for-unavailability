bs=[]
ps=[]
re=[]
# with open('D:\sa_h_1.py\partresult_S=16.txt', 'r') as f:
#     #print(f.readline())
#     for i in range(1500):
#         if i%3==1:
#             bs.append(int(f.readline()))
#         if i%3==2:
#             ps.append(int(f.readline()))
#         if i%3==0:
#             re.append(int(f.readline()))

for i,line in enumerate(open('D:\sa_h_1.py\partresult_S=16.txt', 'r')):
    if i%3==0:
        re.append(float(line.rstrip('\n')))
    if i%3==1:
        bs.append(float(line.rstrip('\n')))
    if i%3==2:
        ps.append(float(line.rstrip('\n')))
#print(ps[499])
bs_ave=sum(bs)/500
ps_ave=sum(ps)/500
print(bs_ave,ps_ave)
#print((bs_ave-ps_ave)/bs_ave)
print(sum(re)/500)
