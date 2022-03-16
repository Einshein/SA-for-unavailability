from encodings import utf_8
import sa_h_1_workload_rcvt
import time
import io

t0 = time.clock()
trails=10
reduced_Una_WlfpToBs=[]
reduced_Una_WlfprtToBs=[]

# under the wlfprt situation
max_Una_Bs=[]
max_Una_Wlfp=[]
max_Una_Wlfprt=[]

for i in range(trails):
    t1=time.clock()
    result=sa_h_1_workload_rcvt.main(16) # main(capacity_range,lambdas)
    max_Una_Bs.append(result[0]) 
    max_Una_Wlfp.append(result[1])
    max_Una_Wlfprt.append(result[2])
    reduced_Una_WlfpToBs.append(result[3])
    reduced_Una_WlfprtToBs.append(result[4])
    print("runtime for this trial:",time.clock()-t1)


max_Una_Bs_Ave=sum(max_Una_Bs)/trails
max_Una_Wlfp_Ave=sum(max_Una_Wlfp)/trails
max_Una_Wlfprt_Ave=sum(max_Una_Wlfprt)/trails
reduced_Una_WlfpToBs_Ave=sum(reduced_Una_WlfpToBs)/trails
reduced_Una_WlfprtToBs_Ave=sum(reduced_Una_WlfprtToBs)/trails

print("Total runtime:", time.clock() - t0)

print("average maximum unavailability of baseline model")
print(max_Una_Bs_Ave)
print("average maximum unavailability of model wlfp:")
print(max_Una_Wlfp_Ave)
print("average maximum unavailability of model wlfprt:")
print(max_Una_Wlfprt_Ave)
print("average reduction in unavailability wlfp to bs:")
print(reduced_Una_WlfpToBs_Ave)
print("average reduction in unavailability wlfprt to bs:")
print(reduced_Una_WlfprtToBs_Ave)



# with io.open('result.txt','a', encoding='utf_8') as f:
#     f.write('delta^-1=90'.decode('utf8')+'\n')  # change it every time after changing the reletive value
#     f.write(str(reduced_una_ave).decode('utf8')+'\n')
#     f.write(str(max_una_bs_ave).decode('utf8')+'\n')
#     f.write(str(max_una_wl_ave).decode('utf8')+'\n')