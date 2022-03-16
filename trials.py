from encodings import utf_8
import sa_h_1_workload
import time
import io

t0 = time.clock()
trails=10
reduced_una=[]
max_una_bs=[]
max_una_wl=[]
for i in range(trails):
    t1=time.clock()
    result=sa_h_1_workload.main(16) # main(capacity_range,lambdas)
    reduced_una.append(result[0]) 
    max_una_bs.append(result[1])
    max_una_wl.append(result[2])
    print("runtime for this trial:",time.clock()-t1)

reduced_una_ave=sum(reduced_una)/trails
max_una_bs_ave=sum(max_una_bs)/trails
max_una_wl_ave=sum(max_una_wl)/trails
print("runtime:", time.clock() - t0)

print("average reduction in unavailability:")
print(reduced_una_ave)
print("average maximum unavailability of baseline model:")
print(max_una_bs_ave)
print("average maximum unavailability of proposed model:")
print(max_una_wl_ave)



# with io.open('result.txt','a', encoding='utf_8') as f:
#     f.write('delta^-1=90'.decode('utf8')+'\n')  # change it every time after changing the reletive value
#     f.write(str(reduced_una_ave).decode('utf8')+'\n')
#     f.write(str(max_una_bs_ave).decode('utf8')+'\n')
#     f.write(str(max_una_wl_ave).decode('utf8')+'\n')