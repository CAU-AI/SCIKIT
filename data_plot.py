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

new_data = data.T

for k in range(0,376):
    print(k)
    y = []
    a = []
    b = []
    aa = k
    t = np.array(range(1,71))
    for i in range(70):
        y.append(data[i][aa])
    z = np.array(y)

    q = np.array(range(71,141))
    for i in range(71,141):
        a.append(data[i][aa])
    w = np.array(a)

    p = np.array(range(141,210))
    for i in range(140,210):
        b.append(data[i][aa])
    i = np.array(b)

    plt.figure()
    plt.plot(t, z, "r.")
    plt.plot(t, w, "b.")
    plt.plot(t, i, "g.")
    plt.show()
"""
scaler = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0).fit(data.astype(float))
data = scaler.transform(data.astype(float))
aaa = [4, 7, 10, 21, 22, 27, 33, 37, 39, 45, 51, 52, 57, 61, 63, 67, 69, 73,72,  75, 79, 78, 93, 223,224, 225, 226, 227,228,229,230,231,232,233,234,235,263,264,265,266,267,302,303,304,305,306,307,308,309,310,311,312,313,327,338,
370,374,344]
aaa.sort()
for k in aaa:
    print(k)
    y = []
    a = []
    b = []
    aa = k
    t = np.array(range(1,71))
    for i in range(70):
        y.append(data[i][aa])
    z = np.array(y)

    q = np.array(range(71, 141))
    for i in range(71, 141):
        a.append(data[i][aa])
    w = np.array(a)

    p = np.array(range(141, 210))
    for i in range(140, 210):
        b.append(data[i][aa])
    i = np.array(b)
    
    plt.figure()
    plt.plot(t,z,"r.")
    plt.plot(t,w,"b.")
    plt.plot(t,i,"g.")
    plt.show()
"""

