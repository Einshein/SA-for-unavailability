from encodings import utf_8
import sa_h_1_workload
import time
import io
import datetime
t0 = time.clock()
trails=500
max_una_wl=[]
max_una_bound=[]
com_time_wl=[]
com_time_bound=[]

for i in range(trails):
    print('trial_number', i)
    t1=time.clock()
    result=sa_h_1_workload.main(16) # main(capacity_range,lambdas)
    max_una_wl.append(result[0])
    max_una_bound.append(result[1])
    com_time_wl.append(result[2])
    com_time_bound.append(result[3])

    print(result[0],result[1],result[2],result[3])
    print("runtime for this trial:",time.clock()-t1)
    print("current time:",datetime.datetime.now())

max_una_wl_ave=sum(max_una_wl)/trails
max_una_bound_ave=sum(max_una_bound)/trails
com_time_wl_ave=sum(com_time_wl)/trails
com_time_bound_ave=sum(com_time_bound)/trails


print("over****************************")
print("runtime:", time.clock() - t0)

print("average maximum unavailability of workload-dependent model:")
print(max_una_wl_ave)
print("average maximum unavailability of the model that use PB:")
print(max_una_bound_ave)
print("average computation time of workload-dependent model:")
print(com_time_wl_ave)
print("average computation time of the model that use PB:")
print(com_time_bound_ave)



with io.open('result_mixedSA.txt','a', encoding='utf_8') as f:
    #f.write('S = 16'.decode('utf8')+'\n')  # change it every time after changing the reletive value for collecting data
    f.write("result:".decode('utf8')+'\n')
    f.write(str(max_una_wl_ave).decode('utf8')+'\n')
    f.write(str(max_una_bound_ave).decode('utf8')+'\n')
    f.write(str(com_time_wl_ave).decode('utf8')+'\n')
    f.write(str(com_time_bound_ave).decode('utf8')+'\n')
