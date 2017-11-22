import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

xl = pd.ExcelFile("Breastcancer_train.xlsx")
#xl = pd.ExcelFile("3D_handwriting_train.xlsx")
#xl = pd.ExcelFile("Music_style_train.xlsx")    NAN data로 인해 오류

data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)
data = np.array(data_excel.values)              #583*9
answer = np.array(answer_excel.values).flatten().transpose()

cov_data = np.cov(data.T)      #9*9
eigenvalue = np.linalg.eigvals(cov_data)     #9

a=eigenvalue.tolist()
b=eigenvalue.tolist()

a.sort(reverse=True)

first_eigenvector_index = b.index(a[1])
second_eigenvector_index = b.index(a[1])
third_eigenvector_index = b.index(a[2])

first_eigenvector = np.empty(1)
second_eigenvector = np.array(1)
third_eigenvector = np.empty(1)

for i in range(0, len(cov_data)):
    first_eigenvector = np.c_[first_eigenvector, [cov_data[i][first_eigenvector_index]]]       #1*10
    second_eigenvector = np.c_[second_eigenvector, [cov_data[i][second_eigenvector_index]]]
    third_eigenvector = np.c_[third_eigenvector, [cov_data[i][third_eigenvector_index]]]

# 42~48 : 첫번째 index 0을 지운후 다시 array로 설정 -> 빈 ndarray 생성이 안돼 추가했습니다.
first_list = first_eigenvector.tolist()     #1*9
second_list = second_eigenvector.tolist()
third_list = third_eigenvector.tolist()

del first_list[0][0]
del second_list[0][0]
del third_list[0][0]

first_eigenvector = np.array(first_list).T
second_eigenvector = np.array(second_list).T
third_eigenvector = np.array(third_list).T


new_data1 = np.dot(data,first_eigenvector)
new_data2 = np.dot(data,second_eigenvector)             #583*1
new_data3 = np.dot(data,third_eigenvector)




fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')


for i in range(0, new_data1.size):
    x=new_data1[i][0]
    y=new_data2[i][0]
    #z=new_data3[i][0]
    z=np.array(answer)[i]
    if z == 1:
        ax.scatter(x, y, z, edgecolors='red')
    else :
        ax.scatter(x, y, z, edgecolors='blue')
plt.show()



