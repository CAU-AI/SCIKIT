from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


xl = pd.ExcelFile("Music_style_train.xlsx")
data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)
data = np.array(data_excel.values)
answer = np.array(answer_excel.values).flatten().transpose()

#plt.plot(range(1,len(data[0])+1),data[0],'ro')
fig = plt.figure()
#ax1 = fig.add_subplot(2,1,1)
#ax2 = fig.add_subplot(2,1,2)
'''

ax=[]
y=[]
x = range(1,len(data[0])+1)
for i in range(1,4):
    ax.append(fig.add_subplot(len(range(1,4)),1,i))

    #y1 = data[0]
    #y2 = data[20]
    y.append(data[i])
    ax[i-1].plot(x,y[i-1],'ro')
'''

x= range(1,len(data[0])+1)
ax1 = fig.add_subplot(3,1,1)
ax2=  fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

plt.show()
