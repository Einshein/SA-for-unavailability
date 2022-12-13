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





# x=['0.00','0.01', '0.02', '0.03', '0.04', '0.05', '0.06','0.07','0.08','0.09']
# #full eu
# y1=[0.014197685262137426, 0.014197685262137426, 0.014197685262137426, 0.014197685262137426, 0.014197685262137426, 0.014197685262137426, 0.014197685262137426, 0.014197685262137426, 0.014197685262137426, 0.014197685262137426]
# #full pb
# y2=[0.014197685262137426, 0.01417619030102387, 0.01429533125371889, 0.014152139115389244, 0.014242976191188634, 0.014550208315828804, 0.014790977100056193, 0.015097992369050352, 0.015769109755344346, 0.01691381645246325]
# #hybrid
# y3=[0.014197685262137426, 0.0140215295198, 0.0141593700212, 0.0141778580041, 0.0144610328487, 0.014457316396212078, 0.01467081004, 0.015038860796, 0.01580763988226292, 0.0167587229168]
# fig=plt.figure()
# ax1 = fig.add_subplot(111)
# plt.xticks(rotation = 0) # Rotates X-Axis Ticks by 45-degrees

# ax1.set(title='', ylabel='Maximum unavailability',xlabel='Ratio of the iterations using PB method to all iterations in SA-MIX, $\it{x}$', yticks=[0.014,0.015,0.016,0.017,0.018,0.019,0.020,0.021,0.022]) 
# plt.show()


x=['0.00','0.01', '0.02', '0.03', '0.04', '0.05', '0.06','0.07','0.08','0.09']
#full eu
y1=[0.014197685262137426, 0.014197685262137426, 0.014197685262137426, 0.014197685262137426, 0.014197685262137426, 0.014197685262137426, 0.014197685262137426, 0.014197685262137426, 0.014197685262137426, 0.014197685262137426]
#full pb
#y2=[0.014197685262137426, 0.01417619030102387, 0.01429533125371889, 0.014152139115389244, 0.014242976191188634, 0.014550208315828804, 0.014790977100056193, 0.015097992369050352, 0.015769109755344346, 0.01691381645246325]
y2=[0.014197685262137426, 0.01417619030102387, 0.01429533125371889, 0.014152139115389244, 0.014633930846467054, 0.014550208315828804, 0.014790977100056193, 0.015097992369050352, 0.015769109755344346, 0.01691381645246325]

#hybrid
y3=[0.014197685262137426, 0.0140215295198, 0.0141593700212, 0.0141778580041, 0.0144610328487, 0.014457316396212078, 0.01467081004, 0.015038860796, 0.01580763988226292, 0.0167587229168]

fig1 = plt.figure()    # 
ax = fig1.add_subplot(111) 
ax.set(title='', yticks=[0.014,0.015, 0.016, 0.017, 0.018, 0.019, 0.020, 0.021, 0.022], xlabel='Ratio of the second stage') 
plt.ylabel(ylabel='Maximum unavailability', fontsize=16)
plt.xlabel(xlabel='Ratio of the second stage', fontsize=16)
plt.tick_params(labelsize=16)
# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
plt.plot(x, y1, marker='s', clip_on=False, label='Full EU') # draw y1
plt.plot(x, y2, marker='o', clip_on=False,label='Full PB') # draw y2
plt.plot(x, y3, marker='p', clip_on=False,label='Hybrid') # draw y2

plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
plt.legend(fontsize=14)
#plt.show()

##################################
x=['0.00','0.01', '0.02', '0.03', '0.04', '0.05', '0.06','0.07','0.08','0.09']
#full eu
y1=[ 7.098285084, 7.098285084, 7.098285084, 7.098285084, 7.098285084, 7.098285084, 7.098285084, 7.098285084, 7.098285084, 7.098285084 ]
#full pb
#y2=[ 7.098285084, 6.471756154, 6.283840318, 5.057835962, 4.563889214, 3.674441746, 3.031066902, 2.37998858, 1.73343329, 1.036746862]
y2=[ 7.098285084, 6.471756154, 6.283840318, 5.057835962, 4.490905700, 3.674441746, 3.031066902, 2.37998858, 1.73343329, 1.036746862]

#hybrid
y3=[ 7.098285084, 7.168200414, 6.890498386, 6.71077571, 6.636278552, 6.628835094, 6.44776276, 6.307170258, 6.295219626, 6.076585826]

fig1 = plt.figure()    # 
ax = fig1.add_subplot(111) 
ax.set(title='', yticks=[1,2,3,4,5,6,7,8], xlabel='Ratio of the second stage') 
plt.ylabel(ylabel='Computation time', fontsize=16)
plt.xlabel(xlabel='Ratio of the second stage', fontsize=16)
plt.tick_params(labelsize=16)
# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
plt.plot(x, y1, marker='s', clip_on=False, label='Full EU') # draw y1
plt.plot(x, y2, marker='o', clip_on=False,label='Full PB') # draw y2
plt.plot(x, y3, marker='p', clip_on=False,label='Hybrid') # draw y2

plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
plt.legend(fontsize=14)
plt.show()







##########################################################

# #x = np.arange(14,19) 
# #variant: M
# x = np.linspace(14, 18, 5)
# y1 = [0.020951277847667402, 0.020214279762985406, 0.019665077232690186, 0.0193478759459687, 0.019186880499269945]      # 曲线 y1
# y2 = [0.016240454380970833, 0.015010432324019125, 0.014005265003568385, 0.013412589310829184, 0.013062404717336818]     # 曲线 y2
# fig1 = plt.figure()    # 
# ax = fig1.add_subplot(111) 
# ax.set(title='', ylabel='Maximum unavailability',xlabel='Upper bound of protection capacity, M',xlim=[14,18],xticks=[14,15,16,17,18],ylim=[0.013,0.021]) 
# plt.ylabel(ylabel='Maximum unavailability', fontsize=16)
# plt.xlabel(xlabel='Upper bound of protection capacity, '+r'$M$', fontsize=16)
# plt.tick_params(labelsize=16)
# # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
# plt.plot(x, y1, marker='s', clip_on=False, label='Baseline model') # draw y1
# plt.plot(x, y2, marker='o', clip_on=False,label='Proposed model') # draw y2
# plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
# plt.legend(fontsize=14)
# #plt.show()
# qq=[]
# q=[]

# for i in range(0,5):
#     q.append((y1[i]-y2[i])/y1[i])
#     qq.append((y1[i]-y2[i])/y1[i])

# print("ex1")
# print("y1",y1)
# print("y2",y2)
# print("reduced una",q)
# print(1,sum(q)/5)



