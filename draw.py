import matplotlib.pyplot as plt   
import numpy as np                
import matplotlib.ticker as mtick


class MathTextSciFormatter(mtick.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)
##########################################################

#x = np.arange(14,19) 
#variant: M
x = np.linspace(14, 18, 5)
y1 = [0.020951277847667402, 0.020214279762985406, 0.019665077232690186, 0.0193478759459687, 0.019186880499269945]      # 曲线 y1
y2 = [0.016240454380970833, 0.015010432324019125, 0.014005265003568385, 0.013412589310829184, 0.013062404717336818]     # 曲线 y2
fig1 = plt.figure()    # 
ax = fig1.add_subplot(111) 
ax.set(title='', ylabel='Maximum unavailability',xlabel='Upper bound of protection capacity, M',xlim=[14,18],xticks=[14,15,16,17,18],ylim=[0.013,0.021]) 
plt.ylabel(ylabel='Maximum unavailability', fontsize=16)
plt.xlabel(xlabel='Upper bound of protection capacity, '+r'$M$', fontsize=16)
plt.tick_params(labelsize=16)
# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
plt.plot(x, y1, marker='s', clip_on=False, label='Baseline model') # draw y1
plt.plot(x, y2, marker='o', clip_on=False,label='Proposed model') # draw y2
plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
plt.legend(fontsize=14)
#plt.show()
qq=[]
q=[]

for i in range(0,5):
    q.append((y1[i]-y2[i])/y1[i])
    qq.append((y1[i]-y2[i])/y1[i])

print("ex1")
print("y1",y1)
print("y2",y2)
print("reduced una",q)
print(1,sum(q)/5)



####################################

#variant: lambda
x = [2 / 100000, 4 / 100000, 6 / 100000, 8 / 100000, 1 / 10000]
y1 = [0.004553046323036749, 0.008673950778759583, 0.01269623505211162, 0.01628806999145799, 0.019665077232690186]      # 曲线 y1
y2 = [0.0033810333124260467, 0.006160534549073553, 0.00896875056231522, 0.011497322832565536, 0.014005265003568385]     # 曲线 y2
fig2 = plt.figure()   
ax = fig2.add_subplot(111) 
ax.set(title='', ylabel='Maximum unavailability',xlabel='Average failure rate, '+ r'$\lambda$', xlim=[2 / 100000,1 / 10000], xticks=[2 / 100000, 4 / 100000, 6 / 100000, 8 / 100000, 1 / 10000]) 
plt.ylabel(ylabel='Maximum unavailability', fontsize=16)
plt.xlabel(xlabel='Average failure rate, '+ r'$\lambda$', fontsize=16)
plt.tick_params(labelsize=16)

ax.set_ylim(0.003,0.021)
ax.set_yticks([0.003,0.006,0.009,0.012,0.015,0.018,0.021])
plt.xticks(rotation=45)
# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("$10^{%d}$"))
# ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("$10^{%d}$"))
plt.plot(x, y1, marker='s', clip_on=False, label='Baseline model') 
plt.plot(x, y2, marker='o', clip_on=False,label='Proposed model') 

plt.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))

plt.legend(fontsize=14)
#plt.show()
q=[]
for i in range(0,5):
    q.append((y1[i]-y2[i])/y1[i])
    qq.append((y1[i]-y2[i])/y1[i])

print("ex2")
print("y1",y1)
print("y2",y2)
print("reduced una",q)
print(2,sum(q)/5)

###########################################################

#variant: mu
x = [2 / 10000, 4 / 10000, 6 / 10000, 8 / 10000, 1 / 1000]
y1 = [0.12572817084460117, 0.04946355449877604, 0.030125483177730318, 0.023065413160206452, 0.019665077232690186]      # 曲线 y1
y2 = [0.10092528492221756, 0.030323883619878058, 0.01866052334821427, 0.014875131797331693, 0.014005265003568385]     # 曲线 y2
fig2 = plt.figure()    
ax = fig2.add_subplot(111) 
ax.set(title='', ylabel='Maximum unavailability',xlabel='Average repair rate, '+ r'$\mu_h$', xlim=[2 / 10000,1 / 1000],xticks=[2 / 10000, 4 / 10000, 6 / 10000, 8 / 10000, 1 / 1000]) 
plt.ylabel(ylabel='Maximum unavailability', fontsize=16)
plt.xlabel(xlabel='Average repair rate, '+ r'$\mu_h$', fontsize=16)
plt.tick_params(labelsize=16)

ax.set_ylim(0.01,0.13)
ax.set_yticks([0.01,0.04,0.07,0.1,0.13])
plt.xticks(rotation=45)
# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
# ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
plt.plot(x, y1, marker='s', clip_on=False, label='Baseline model')
plt.plot(x, y2, marker='o', clip_on=False,label='Proposed model') 
plt.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))

plt.legend(fontsize=14)
#plt.show()
q=[]
for i in range(0,5):
    q.append((y1[i]-y2[i])/y1[i])
    qq.append((y1[i]-y2[i])/y1[i])

print("ex3")
print("y1",y1)
print("y2",y2)
print("reduced una",q)
print(1,sum(q)/5)

###########################################################################

#variant: Upper bound of recovery time U
x = [80, 90, 100, 110, 120]
y1 = [0.01772628409910509, 0.018709670433119004, 0.019665077232690186, 0.020648659590137376, 0.02170472007951516]      # 曲线 y1
y2 = [0.012351916990485066, 0.013176336982278115, 0.014005265003568385, 0.014948330703372353, 0.01584818559333616]    # 曲线 y2
fig2 = plt.figure()    # 定义一个图像窗口
ax = fig2.add_subplot(111) 
ax.set(title='', ylabel='Maximum unavailability',xlabel='Upper bound of recovery time, '+ r'U', xlim=[80, 120],xticks=[80, 90, 100, 110, 120]) 
plt.ylabel(ylabel='Maximum unavailability', fontsize=16)
plt.xlabel(xlabel='Upper bound of recovery time, '+ r'$U$', fontsize=16)
plt.tick_params(labelsize=16)

ax.set_ylim(0.012,0.022)
ax.set_yticks([0.012,0.014,0.016,0.018,0.020,0.022])
# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
plt.plot(x, y1, marker='s', clip_on=False, label='Baseline model') 
plt.plot(x, y2, marker='o', clip_on=False,label='Proposed model') 
plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))

plt.legend(fontsize=14)
#plt.show()
q=[]
for i in range(0,5):
    q.append((y1[i]-y2[i])/y1[i])
    qq.append((y1[i]-y2[i])/y1[i])

print("ex4")
print("y1",y1)
print("y2",y2)
print("reduced una",q)
print(1,sum(q)/5)


##########################################################################

#variant: F
x = [80, 90, 100, 110, 120]
y1 = [0.0185457663425, 0.019114604017, 0.019665077232690186, 0.0201769428754, 0.0209509951974]      # 曲线 y1
y2 = [0.0122915482034, 0.0131122548054, 0.014005265003568385, 0.0150721000711, 0.0165078963272]    # 曲线 y2
fig2 = plt.figure()    # 定义一个图像窗口
ax = fig2.add_subplot(111) 
ax.set(title='', ylabel='Maximum unavailability',xlabel='Number of functions, '+ r'|F|', xlim=[80, 120],xticks=[80, 90, 100, 110, 120]) 
plt.ylabel(ylabel='Maximum unavailability', fontsize=16)
plt.xlabel(xlabel='Number of functions, '+ r'$|F|$', fontsize=16)
plt.tick_params(labelsize=16)

ax.set_ylim(0.012,0.021)
ax.set_yticks([0.012,0.014,0.016,0.018,0.020,0.022])
# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
plt.plot(x, y1, marker='s', clip_on=False, label='Baseline model') 
plt.plot(x, y2, marker='o', clip_on=False,label='Proposed model') 
plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))

plt.legend(fontsize=14)
#plt.show()
q=[]
for i in range(0,5):
    q.append((y1[i]-y2[i])/y1[i])
    qq.append((y1[i]-y2[i])/y1[i])

print("ex5")
print("y1",y1)
print("y2",y2)
print("reduced una",q)
print(1,sum(q)/5)


######################################################

#variant: S, number of backup servers
x = [16, 18, 20, 22, 24]
y1 = [0.021844459772844074, 0.0201234171985, 0.019665077232690186, 0.019687300302751655, 0.01949377992449695]      # 曲线 y1
y2 = [0.01807094437587707, 0.0155505499058, 0.014005265003568385, 0.013444385795921698, 0.0129479049910944]     # 曲线 y2
fig1 = plt.figure()    # 
ax = fig1.add_subplot(111) 
ax.set(title='', ylabel='Maximum unavailability',xlabel='Number of backup servers, |S|',xlim=[16,24],xticks=[16,18,20,22,24]) 
plt.ylabel(ylabel='Maximum unavailability', fontsize=16)
plt.xlabel(xlabel='Number of backup servers, '+r'$|S|$', fontsize=16)
plt.tick_params(labelsize=16)

ax.set_ylim(0.012,0.022)
ax.set_yticks([0.012,0.014,0.016,0.018,0.020,0.022])

# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
plt.plot(x, y1, marker='s', clip_on=False, label='Baseline model') # draw y1
plt.plot(x, y2, marker='o', clip_on=False,label='Proposed model') # draw y2
plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
plt.legend(fontsize=14)
#plt.show()
q=[]
for i in range(0,5):
    q.append((y1[i]-y2[i])/y1[i])
    qq.append((y1[i]-y2[i])/y1[i])

print("ex6")
print("y1",y1)
print("y2",y2)
print("reduced una",q)
print(1,sum(q)/5)

print("total reduced una:", sum(qq)/30)

plt.show()