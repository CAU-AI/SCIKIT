from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


xl = pd.ExcelFile("Music_style_train.xlsx")
data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)

xl_2 = pd.ExcelFile("demo_test2.xlsx")
demo_excel = pd.read_excel(io=xl_2, sheetname=0, header=None)
demo_answer_excel = pd.read_excel(io=xl_2, sheetname=1, header=None)
f = open('result2.txt', 'w+t')

data = np.array(data_excel.values)
answer = np.array(answer_excel.values).flatten().transpose()
demo_data = np.array(demo_excel.values)
demo_answer = np.array(demo_answer_excel.values).flatten().transpose()

scaler = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0).fit(data.astype(float))
data = scaler.transform(data.astype(float))
scaler = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0).fit(demo_data.astype(float))
demo_data = scaler.transform(demo_data.astype(float))

transpose_data = data.T
transpose_demo_data = demo_data.T
check_data = []
check_demo_data = []

#a = [4, 7, 10, 21, 22, 27, 33, 37, 39, 45, 51, 52, 57, 61, 63, 67, 69, 73,72,75,79, 78, 93, 223,224, 225, 226, 227,228,229,230,231,232,233,234,235,263,264,265,266,267,302,303,304,305,306,307,308,309,310,311,312,313,327,338,344]
#a.sort()

a= [7,21,27,33,45,51,52,69,72,73,79]   #최종본
#a = [72,73,75,78,79,344,355,356]

for i in a:
    check_data.append(transpose_data[i])
aa = np.array(check_data)
data = aa.T

sub_data = data[:70]
sub_data = np.append(sub_data, data[140:210], axis=0)
sub_answer = answer[0:70]
sub_answer = np.append(sub_answer, answer[140:210], axis=0)

for i in a:
    check_demo_data.append(transpose_demo_data[i])
bb = np.array(check_demo_data)
demo_data = bb.T

print(data.shape)
print(sub_data.shape)
print(demo_data.shape)

nbrs = KNeighborsClassifier(n_neighbors=7)
train_model = nbrs.fit(data, answer)
demo_pred = train_model.predict(demo_data)
j = 0
for i in demo_pred:
    if i == '리드미컬':
        j+=1
        continue
    else:
        gnb = GaussianNB()
        train_model = gnb.fit(sub_data, sub_answer)
        temp_data = demo_data[j]
        temp_data = temp_data.reshape(1, -1)
        temp = train_model.predict(temp_data)
        demo_pred[j] = temp[0]
        j+=1

print(demo_pred)

for i in range(len(demo_pred)):
    f.write(str(demo_pred[i]) + '\n')


