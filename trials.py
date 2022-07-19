from encodings import utf_8
import sa_h_1_workload
import time
import io
import datetime
t0 = time.clock()
trails=500
max_una=[]
com_time=[]

for i in range(trails):
    print('trial_number', i)
    t1=time.clock()
    result=sa_h_1_workload.main(16) # main(capacity_range)
    max_una.append(result[0])
    com_time.append(result[1])

    print(result[0],result[1])
    print("runtime for this trial:",time.clock()-t1)
    print("current time:",datetime.datetime.now())

max_una_ave=sum(max_una)/trails
com_time_ave=sum(com_time)/trails


print("over****************************")
print("runtime:", time.clock() - t0)

print("average maximum unavailability:")
print(max_una_ave)
print("average computation time:")
print(com_time_ave)


with io.open('result_mixedSA.txt','a', encoding='utf_8') as f:
    f.write("***********************".decode('utf8')+'\n')
    f.write("pb:0.5".decode('utf8')+'\n')
    f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())).decode('utf8')+'\n')
    f.write("trails number:".decode('utf8')+str(trails).decode('utf8')+'\n')
    f.write("result:".decode('utf8')+'\n')
    f.write(str(max_una_ave).decode('utf8')+'\n')
    f.write(str(com_time_ave).decode('utf8')+'\n')

