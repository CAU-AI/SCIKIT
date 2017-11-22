from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

xl = pd.ExcelFile("Music_style_train.xlsx")
data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)
data = np.array(data_excel.values)                                  #210 * 376
answer = np.array(answer_excel.values).flatten().transpose()

#0~68, 69~139, 140~209

def e(a,b):
	E = []
	temp = 0.0
	for j in range(0,len(data[0])):
		for i in range(a, b+1):
			temp += data[i,j]
		temp = temp/(b-a)
		E.append(temp)
		temp = 0
	return E




fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
#ax2 = fig.add_subplot(2,1,2)
ax=[]
y=[]
x = range(1,len(data[0])+1)


#for i in range(100,103):
ax.append(fig.add_subplot(1,1,1))
#ax.append(fig.add_subplot(3,1,2))
#ax.append(fig.add_subplot(3,1,3))
y.append(e(0,69))
#y.append(e(69,140))
#y.append(e(140,210))
#y1 = data[0]
#y2 = data[20]
ax[0].plot(x,y[0],'ro')
#ax[1].plot(x,y[1],'ro')
#ax[2].plot(x,y[2],'ro')

def nice(a,b):
    list = e(a,b)
    for i in range(a,b):
        if(np.isnan(list[i])):
            list[i] = 1
    origin_list = list[:]
    list.sort()
    list_index = []
    for i in range(b-1,b-11,-1):
        list_index.append(origin_list.index(list[i]))

    print(list_index)
    return list_index


nice(0,69)
nice(69,140)
nice(140,209)


plt.show()