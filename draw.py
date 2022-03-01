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


#x = np.arange(14,19) 
#variant: M
x = np.linspace(14, 18, 5)
y1 = [0.02100215765799985, 0.020651076805796576, 0.02055638706548533, 0.020132600465071794, 0.019993571816002278]      # 曲线 y1
y2 = [0.01735280235993259, 0.01665477081253022, 0.016145208369117085, 0.01584259324009574, 0.015536404712971706]     # 曲线 y2
fig1 = plt.figure()    # 
ax = fig1.add_subplot(111) 
ax.set(title='', ylabel='Maximum unavailability',xlabel='Upper bound of protection capacity, M',xlim=[14,18],xticks=[14,15,16,17,18],ylim=[0.015,0.022]) 
plt.ylabel(ylabel='Maximum unavailability', fontsize=16)
plt.xlabel(xlabel='Upper bound of protection capacity, '+r'$M$', fontsize=16)
plt.tick_params(labelsize=16)
# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1E'))
plt.plot(x, y1, marker='s', clip_on=False, label='Baseline model') # draw y1
plt.plot(x, y2, marker='o', clip_on=False,label='Proposed model') # draw y2
plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
plt.legend(fontsize=14)
#plt.show()
q=[]
for i in range(0,5):
    q.append((y1[0]-y2[0])/y1[0])
print(1,sum(q)/5)

qq=[]
for i in range(0,5):
    qq.append((y1[0]-y2[0])/y1[0])




#variant: lambda
x = [2 / 100000, 4 / 100000, 6 / 100000, 8 / 100000, 1 / 10000]
y1 = [0.0046627326345636474, 0.00896018065198039, 0.012938921128644611, 0.01672790605513416, 0.02055638706548533]      # 曲线 y1
y2 = [0.0038372353225161367, 0.007231186448063089, 0.010263445250794551, 0.013152605421676648, 0.016145208369117085]     # 曲线 y2
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
    q.append((y1[0]-y2[0])/y1[0])
print(2,sum(q)/5)
for i in range(0,5):
    qq.append((y1[0]-y2[0])/y1[0])



#variant: mu
x = [2 / 10000, 4 / 10000, 6 / 10000, 8 / 10000, 1 / 1000]
y1 = [0.13941302736445854, 0.056527917010649416, 0.03345843369047702, 0.025344471377281214, 0.02055638706548533]      # 曲线 y1
y2 = [0.12984885487566655, 0.04718883468663414, 0.026614833539103332, 0.02026312700840011, 0.016145208369117085]     # 曲线 y2
fig2 = plt.figure()    
ax = fig2.add_subplot(111) 
ax.set(title='', ylabel='Maximum unavailability',xlabel='Average repair rate, '+ r'$\mu_h$', xlim=[2 / 10000,1 / 1000],xticks=[2 / 10000, 4 / 10000, 6 / 10000, 8 / 10000, 1 / 1000]) 
plt.ylabel(ylabel='Maximum unavailability', fontsize=16)
plt.xlabel(xlabel='Average repair rate, '+ r'$\mu_h$', fontsize=16)
plt.tick_params(labelsize=16)

ax.set_ylim(0.01,0.15)
ax.set_yticks([0.01,0.08,0.15])
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
    q.append((y1[0]-y2[0])/y1[0])
print(3,sum(q)/5)
for i in range(0,5):
    qq.append((y1[0]-y2[0])/y1[0])



#variant: Upper bound of recovery time U
x = [80, 90, 100, 110, 120]
y1 = [0.019144567175345722, 0.019990717583343785, 0.02055638706548533, 0.022215407812363145, 0.023217181980719193]      # 曲线 y1
y2 = [0.015452283833569138, 0.016165836680500333, 0.016145208369117085, 0.018593713555951526, 0.019194029703872077]    # 曲线 y2
fig2 = plt.figure()    # 定义一个图像窗口
ax = fig2.add_subplot(111) 
ax.set(title='', ylabel='Maximum unavailability',xlabel='Upper bound of recovery time, '+ r'U', xlim=[80, 120],xticks=[80, 90, 100, 110, 120]) 
plt.ylabel(ylabel='Maximum unavailability', fontsize=16)
plt.xlabel(xlabel='Upper bound of recovery time, '+ r'$U$', fontsize=16)
plt.tick_params(labelsize=16)

ax.set_ylim(0.014,0.024)
ax.set_yticks([0.014,0.016,0.018,0.02,0.022,0.024])
# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
plt.plot(x, y1, marker='s', clip_on=False, label='Baseline model') 
plt.plot(x, y2, marker='o', clip_on=False,label='Proposed model') 
plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))

plt.legend(fontsize=14)
plt.show()
q=[]
for i in range(0,5):
    q.append((y1[0]-y2[0])/y1[0])
print(4,sum(q)/5)

for i in range(0,5):
    qq.append((y1[0]-y2[0])/y1[0])
print("total:", sum(qq)/20)

