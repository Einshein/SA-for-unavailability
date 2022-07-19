#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division
import random
import math
import string
import time
import numpy as np


def sa(F, S, c_j, r_j, lambdas, mu, delta_j, T_i, T_t, cool):  # c_j, r_j, delta_j : list
    allocation = []
    for j in range(S):
        allocation.append(0)  

    #　randomly allocate the backup server to each function in F as the initial allocation
    for i in range(F):
        flag = 1
        while(flag):
            j = random.randint(0,S-1)
            if c_j[j] - allocation[j] > 0:
                allocation[j] = allocation[j] + 1
                flag = 0
    
    P_j = []     #p_j[]: the PB for all servers/components       
    P = 0
    #iterate servers, get the unavailability of the component j in the current allocation
    for j in range(S):
        P_jj = costf_pb(c_j[j], r_j[j], lambdas, mu, delta_j[j], allocation[j])
        P_j.append(P_jj)
        if P_jj > P:
            P = P_jj
    #   
    
    # P is assumed to be the (minimized) maximum PB of the current allocation
    results = [P, allocation] 
    #print(results)
    ###############################

    T = T_i  # initial temp  
    # numberOfIterations=13802
    itr=0
    while T > T_t:
        itr=itr+1

        #numberOfIterations=numberOfIterations+1
        T = cool*T       #temperature reduction
        print("temp:",T)
        allocation1 = []  # a new allocation
        for j in range(S):
            allocation1.append(allocation[j])    #deep copy allocation to allocation1
            
        flag = 1
        while(flag):
            # randomly select 2 servers
            j1 = random.randint(0,S-1)
            j2 = random.randint(0,S-1)

            # Assign a function protected by j1 to j2 to randomly move
            if j1 != j2 and allocation1[j1] > 0 and allocation1[j2] < c_j[j2]:   
                allocation1[j1] = allocation1[j1] - 1
                allocation1[j2] = allocation1[j2] + 1
                flag = 0
                
        #############################
        switch=int(13802*0.5)  #default: 0.5 
        if itr < switch:
            P_j1 = P_j[:]  #P_j1[]: new unavailabilities for all servers/components for the changed allocation
            #update the unavailabilities of j1 and j2
            P_j1[j1] = costf_pb(c_j[j1], r_j[j1], lambdas, mu, delta_j[j1], allocation1[j1]) #
            P_j1[j2] = costf_pb(c_j[j2], r_j[j2], lambdas, mu, delta_j[j2], allocation1[j2]) #

            P1 = 0
            #set P1 to the maximum unavailability in the new P_j1[]
            for j in range(S):
                if P_j1[j] > P1:
                    P1 = P_j1[j]

            #P is the previous maximum unavailability, 
            #if new value J1 is better(lower then J), accept J1
            #or if new J1 is worse, accept it with a probability.
            if (P1 < P or random.random() < pow(math.e, -(P1 - P)/T)):
                allocation = allocation1[:]
                P = P1
                P_j = P_j1[:]
                
                results[0] = P
                results[1] = allocation
            
        if itr == switch:
            # compute both J and P
            # P:
            P_j1 = P_j[:]  #P_j1[]: new unavailabilities for all servers/components for the changed allocation
            #update the unavailabilities of j1 and j2
            P_j1[j1] = costf_pb(c_j[j1], r_j[j1], lambdas, mu, delta_j[j1], allocation1[j1]) #
            P_j1[j2] = costf_pb(c_j[j2], r_j[j2], lambdas, mu, delta_j[j2], allocation1[j2]) #

            P1 = 0
            #set P1 to the maximum unavailability in the new P_j1[]
            for j in range(S):
                if P_j1[j] > P1:
                    P1 = P_j1[j]

            # J (initial):
            J_j = []     #J_j[]: the unavailabilities for all servers/components       
            J = 0
            #iterate servers, get the unavailability of the component j in the current allocation
            for j in range(S):
                J_jj = costf_wl(c_j[j], r_j[j], lambdas, mu, delta_j[j], allocation[j])
                J_j.append(J_jj)
                if J_jj > J:
                    J = J_jj
            # comparison: P
            # set initial J

            if (P1 < P or random.random() < pow(math.e, -(P1 - P)/T)):
                allocation = allocation1[:]

                results[0] = J
                results[1] = allocation

        if itr > switch:
            J_j1 = J_j[:]  #J_j1[]: new unavailabilities for all servers/components for the changed allocation
            #update the unavailabilities of j1 and j2
            J_j1[j1] = costf_wl(c_j[j1], r_j[j1], lambdas, mu, delta_j[j1], allocation1[j1]) #
            J_j1[j2] = costf_wl(c_j[j2], r_j[j2], lambdas, mu, delta_j[j2], allocation1[j2]) #
            
            J1 = 0
            #set J1 to the maximum unavailability in the new allocation J_j1
            for j in range(S):
                if J_j1[j] > J1:
                    J1 = J_j1[j]

            #J is the previous maximum unavailability, 
            #if new value J1 is better(lower then J), accept J1
            #or if new J1 is worse, accept it with a probability.
            if (J1 < J or random.random() < pow(math.e, -(J1 - J)/T)):
                allocation = allocation1[:]
                J = J1
                J_j = J_j1[:]
                results[0] = J
                results[1] = allocation
            #print(results)
    #print("numberOfIterations:",numberOfIterations)
    return results
            

def costf_wl(c_jj, r_jj, lambdas, mu, delta_jj, L_jj): #L_jj: number of functions which are protected by server j
    #numbner of fesible states
    if L_jj >= r_jj:
        num_feast = (r_jj*r_jj + r_jj) / 2 + (r_jj + 1)*(L_jj - r_jj + 1) + L_jj + 1
    else:
        num_feast = (L_jj*L_jj + 5*L_jj + 4) / 2
    
    num_feast = int(num_feast)
    #print(num_feast)
    #listing all states
    all_states = []
    if L_jj >= r_jj:
        for t in range(r_jj):
            for o in range(t + 1):
                state = []
                m = L_jj - t
                n = 0
                p = t - o
                q = 1
                state = [m, n, o, p, q]
                all_states.append(state)
        for o in range(r_jj + 1):
            for n in range(L_jj - r_jj + 1):
                state = []
                m = L_jj - r_jj - n
                p = r_jj - o
                q = 1
                state = [m, n, o, p, q]
                all_states.append(state)
        for n in range(L_jj + 1): #server fail, m or n
            state = []
            m = L_jj - n
            o = 0
            p = 0
            q = 0
            state = [m, n, o, p, q]
            all_states.append(state)
    else:
        for m in range(L_jj + 1):
            for o in range(L_jj - m + 1):
                state = []
                n = 0
                p = L_jj - m - o
                q = 1
                state = [m, n, o, p, q]
                all_states.append(state)
        for n in range(L_jj + 1):
            state = []
            m = L_jj - n
            o = 0
            p = 0
            q = 0
            state = [m, n, o, p, q]
            all_states.append(state)
    #print(all_states)
    # obtain mutiple equations
    A = []
    for k in range(num_feast):
        a = []
        for kk in range(num_feast):
            a.append(0)
        A.append(a)
    
    for k in range(num_feast):#all_states[k]:state(m,n,o,p,q); 0:m 1:n 2:0 3:p 4:q
    #incoming sates    
        #t1 #
        if (all_states[k][0] >= 1 and all_states[k][2] + all_states[k][3] == r_jj) or (all_states[k][0] >= 1 and all_states[k][4] == 0):
            state = [all_states[k][0] - 1, all_states[k][1] + 1, all_states[k][2], all_states[k][3], all_states[k][4]]
            rate = (all_states[k][1] + 1 + all_states[k][2]) * mu
            indexs = all_states.index(state)
            A[k][indexs] = rate
        #t2
        if all_states[k][0] >= 1 and all_states[k][2] + 1 + all_states[k][3] <= r_jj and all_states[k][4] == 1:
            state = [all_states[k][0] - 1, all_states[k][1], all_states[k][2] + 1, all_states[k][3], all_states[k][4]]
            rate = (all_states[k][2] + 1) * mu
            indexs = all_states.index(state)
            A[k][indexs] = rate        
        #t3
        if all_states[k][0] >= 1 and all_states[k][2] + all_states[k][3] + 1 <= r_jj and all_states[k][4] == 1:
            state = [all_states[k][0] - 1, all_states[k][1], all_states[k][2], all_states[k][3] + 1, all_states[k][4]]
            rate = (all_states[k][3] + 1) * mu
            indexs = all_states.index(state)
            A[k][indexs] = rate            
        #t4
        if all_states[k][0] >= 1 and all_states[k][2] >= 1 and all_states[k][2] + all_states[k][3] == r_jj:
            state = [all_states[k][0] - 1, all_states[k][1] + 1, all_states[k][2] - 1, all_states[k][3] + 1, all_states[k][4]]
            rate = (all_states[k][3] + 1) * mu
            indexs = all_states.index(state)
            A[k][indexs] = rate   
        #t5
        if all_states[k][1] >= 1:
            state = [all_states[k][0] + 1, all_states[k][1] - 1, all_states[k][2], all_states[k][3], all_states[k][4]]
            rate = (all_states[k][0] + 1) * lambdas
            indexs = all_states.index(state)
            A[k][indexs] = rate    
        #t6
        if all_states[k][1] == 0 and all_states[k][2] >= 1:
            state = [all_states[k][0] + 1, all_states[k][1], all_states[k][2] - 1, all_states[k][3], all_states[k][4]]
            rate = (all_states[k][0] + 1) * lambdas
            indexs = all_states.index(state)
            A[k][indexs] = rate    
        #t7
        if all_states[k][3] >= 1:
            state = [all_states[k][0], all_states[k][1], all_states[k][2] + 1, all_states[k][3] - 1, all_states[k][4]]
            rate = (all_states[k][2] + 1) * delta_jj
            indexs = all_states.index(state)
            A[k][indexs] = rate        
        #t8 workload
        if all_states[k][4] == 0: #q=0
            l = min(all_states[k][1], r_jj)
            for o1 in range(l + 1):
                p1 = l - o1
                state = [all_states[k][0], all_states[k][1] - o1 - p1, o1, p1, all_states[k][4] + 1]
                rate = lambda_wl((o1+p1)/r_jj) #workload-dependent failure probability
                indexs = all_states.index(state)
                A[k][indexs] = rate                
        #t9
        if all_states[k][3] == 0 and all_states[k][4] == 1:
            state = [all_states[k][0], all_states[k][1] + all_states[k][2], 0, 0, all_states[k][4] - 1]
            rate = mu
            indexs = all_states.index(state)
            A[k][indexs] = rate 

    #outgoing sates        
        rates = 0
        #t10
        if all_states[k][1] >= 1:
            rate = (all_states[k][1] + all_states[k][2]) * mu
            rates = rates - rate       
        #t11
        if all_states[k][1] == 0 and all_states[k][2] >= 1:
            rate = all_states[k][2] * mu
            rates = rates - rate    
        #t12
        if all_states[k][1] == 0 and all_states[k][3] >= 1:
            rate = all_states[k][3] * mu
            rates = rates - rate        
        #t13
        if all_states[k][1] >= 1 and all_states[k][3] >= 1:
            rate = all_states[k][3] * mu
            rates = rates - rate    
        #t14
        if (all_states[k][0] >= 1 and all_states[k][2] + all_states[k][3] == r_jj) or (all_states[k][0] >= 1 and all_states[k][4] == 0):
            rate = all_states[k][0] * lambdas
            rates = rates - rate    
        #t15
        if all_states[k][0] >= 1 and all_states[k][2] + all_states[k][3] + 1 <= r_jj and all_states[k][4] == 1:
            rate = all_states[k][0] * lambdas
            rates = rates - rate                                
        #t16
        if all_states[k][2] >= 1:
            rate = all_states[k][2] * delta_jj
            rates = rates - rate            
        #t17 worklload
        if all_states[k][4] == 1:
            rate = lambda_wl((all_states[k][2]+all_states[k][3])/r_jj)
            rates = rates - rate        
        #t18
        if all_states[k][4] == 0:
            rate = mu
            rates = rates - rate
        A[k][k] = rates
    
    for k in range(num_feast):
        A[num_feast - 1][k] = 1
    
    #print(A)
    
    B = []
    for k in range(num_feast - 1):
        B.append(0)
    B.append(1)
    
    #print(B)
    
    C = np.linalg.solve(A,B)
    
    #print(C)
    
    At = 0
    Ql = 0
    At1 = 0
    Ql1 = 0
    for k in range(num_feast):
        At = At + (all_states[k][1] + all_states[k][2]) * C[k]
        Ql = Ql + (all_states[k][0] * lambdas + all_states[k][3] * all_states[k][4] * lambdas) * C[k]
        At1 = At1 + (all_states[k][0] + all_states[k][3]) * C[k]
        Ql1 = Ql1 + ((all_states[k][1] + all_states[k][2] + all_states[k][3]) * mu + all_states[k][2] * delta_jj) * C[k]
    
    ti_j = At / Ql
    ti1_j = At1 / Ql1
    
    #print(ti_j)
    #print(ti1_j)
    
    Q_jj = ti_j / (ti_j + ti1_j)
    #print(Q_jj)
    
    return Q_jj

# return the bound of the  
def costf_pb(c_jj, r_jj, lambdas, mu, delta_jj, L_jj): # L_jj: number of functions which are protected by server j; lambdas(lambda_h): failure rate of fucntions; lambda_wl(lambda_j):failure rate of servers
    a = lambdas/(lambdas+mu)
    b = delta_jj/(lambda_wl(1)+mu)    # assume that the workload of backup server is 1
    c = mu + delta_jj + lambda_wl(1)

    if L_jj <= r_jj: # case 3
        Q_U = a-b*((a*mu)/(c+(1+b)*mu))
        Q_L = a-b*(((1-a)*lambdas+a*mu)/(c+(1+b)*mu))
    else: # case 4
        z_L = min((r_jj*mu)/(L_jj-r_jj),(mu*r_jj)/L_jj)             # z_i^L
        Q_U = a-b*((a*z_L)/(c+(1+b)*z_L))
        z_U = max((r_jj*mu),mu)
        Q_L = a-b*(((1-a)*lambdas+a*z_U)/(c+(1+b)*z_U))
    
    return 0.9*Q_U+0.1*Q_L
    # return Q_L much worse



# def solve(F, S, c_j, r_j, lambdas, mu, delta_j, T_i, T_t, cool):
#     J = sa(F, S, c_j, r_j, lambdas, mu, delta_j, costf_bs, T_i, T_t, cool)[0]
#     return J
    
#woroload-dependent failure probability of the backup server
def lambda_wl(omega):
    if omega<0 or omega>1:
        return
    l_min = 1 / 1000000
    l_max = 1 / 10000
    return l_min+omega*(l_max-l_min)


#if __name__ == "__main__":
def main(capacity_range): # change it in trials.py; default : 16
#    costf(5, 3, 1 / 100, 1 / 10, 1 / 0.01, 2)
    F = 100  # change it; default:100
    S = 20   # change it; default:20
    
    lambdas = 1 / 10000  # change it; default:1 / 10000
    mu = 1 / 1000  # change it; default:1 / 1000
    #print((1/mu) / (1/mu + 1/lambdas))   #
    #print("sum(c_j) and c_j")
    #print(sum(c_j))
    #print("c_j:",c_j) 
    #print("sum(r_j) and r_j")
    #print(sum(r_j))
    #print("r_j",r_j)
    delta_j = []
    for j in range(S):
        delta_j.append(1 / random.uniform(10,100))    # change it; default:100

    c_j = []
    r_j = []
    while True:
        for j in range(S):  # randomly set the list c_j and r_j
            c_jj = random.randint(1,capacity_range)
            r_jj = random.randint(1,c_jj)
            c_j.append(c_jj)
            r_j.append(r_jj)
        if sum(c_j) >= F: #  insure total protecion capacity >= number of functions (shared)
            break

    T_i = 10000000.0    #D_init
    T_t = 0.00001       #D_term
    cool = 0.998  #default:0.99

    # the optimal allocation J_WL under the workload-dependent scenario
    # print("The optimal allocation J_WL and its maximum unavailability under the workload-dependent scenario:")

    t0=time.clock()
    J_mix = sa(F, S, c_j, r_j, lambdas, mu, delta_j, T_i, T_t, cool)
    t1=time.clock()-t0
    #print(t2)

    return J_mix[0], t1

        

        
    
    