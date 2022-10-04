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





x=['0.0\n(SA-EU)','0.5','0.6','0.7','0.8','0.9','0.95','0.96','0.97','0.98','0.99','1.0\n(SA-PB)']
y1=[28.163934644,14.879518324,12.206499784,9.432058318,7.174580248,3.641886068,1.995819818,1.6236188,1.276133262,0.951333438,0.58341126,0.233341802]
y2=[0.014279978994,0.0142765428713,0.0142227754638,0.0142814319377,0.0142569414516,0.0142505362645,0.014637721712,0.0147068204199,0.0152559482941,0.0157080629584,0.0167428277491,0.0210954541336]
fig=plt.figure()
ax1 = fig.add_subplot(111)
ax1.bar(x,y1,color='lightgray',edgecolor = "black")
plt.xticks(rotation = 0) # Rotates X-Axis Ticks by 45-degrees

ax1.legend(labels=['Computation time [s]'], bbox_to_anchor=(0, 1.08), loc='upper left', borderaxespad=0)
ax2 = ax1.twinx()
ax2.plot(x,y2,color='tab:blue',marker='o', clip_on=False,label='Maximum unavailability')
ax1.set(title='', ylabel='Computation time [s]',xlabel='Ratio of the iterations using PB method to all iterations in SA-MIX, $\it{x}$',yticks=[0,5,10,15,20,25,30]) 
ax2.set(title='', ylabel='Maximum unavailability',yticks=[0.014,0.015,0.016,0.017,0.018,0.019,0.020,0.021,0.022]) 
plt.legend(labels=['Maximum unavailability'], bbox_to_anchor=(1, 1.08), loc='upper right', borderaxespad=0)
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



