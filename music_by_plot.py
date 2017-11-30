from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


xl = pd.ExcelFile("Music_style_train.xlsx")
data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)

data = np.array(data_excel.values)
answer = np.array(answer_excel.values).flatten().transpose()
transpose_data = data.T

check_data = []
#new_data.append(transpose_data[4])

a = [4, 7, 10, 21, 22, 27, 33, 37, 39, 45, 51, 52, 57, 61, 63, 67, 69, 73,72,  75, 79, 78, 93, 223,224, 225, 226, 227,228,229,230,231,232,233,234,235,263,264,265,266,267,302,303,304,305,306,307,308,309,310,311,312,313,327,338,
370,374,344]
a.sort()

#a= [7,21,27,33,45,51,52,72,73,79]   #최종본
#a = [72,73,75,78,79,344,355,356]

"""
#NaN data 처리 방법(1) : NaN data 포함 특징 배제 
count = 0
for i in a:
    for j in transpose_data[i]:
        if np.isnan(j):
            print(i)
"""

for i in a:
    check_data.append(transpose_data[i])

aa = np.array(check_data)
data = aa.T

sub_data = data[:70]
sub_data = np.append(sub_data, data[140:210], axis = 0)
sub_answer = answer[0:70]
sub_answer = np.append(sub_answer, answer[140:210], axis = 0)

"""
# NaN data처리 방법(2) : partial_distance_strategy
new_data = np.zeros((len(data),len(data[0])),dtype="i")
for i in range(len(data)):
    for j in range(len(data[0])):
        if np.isnan(data[i][j]):
            new_data[i][j] = 0
        else:
            new_data[i][j] = 1

train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=0.1)
test_pred_list =[]

for i in range(0, len(test_data)):      #test_data 크기
    distance_range = []
    for j in range(len(train_data)):
        I_value = 0
        a = 0.0
        for k in range(len(train_data[0])):
            I_value += new_data[j][k]
            if np.isnan(train_data[j][k]):
                difference = 0.0
            else:
                difference = abs(train_data[j][k] - test_data[i][k])
            b = difference
            a += b
        distance = len(data[0]) / I_value * a
        distance_range.append(distance)

    label1 = 0
    label2 = 0
    label3 = 0
    origin_distance_range = distance_range[:]
    distance_range.sort()                 # len(distance_range) = 168, len(train_data) = 168, len(test_data) = 42    test_size = 0.2

    for p in range(0, 4):
        index = origin_distance_range.index(distance_range[p])
        if train_answer[index] == '멜랑꼴리':
            label1 += 1
        elif train_answer[index] == '리드미컬':
            label2 += 1
        else:
            label3 += 1
    if max(label1, label2, label3) == label1:
        pred_value = '멜랑꼴리'
    elif max(label1, label2, label3) == label2:
        pred_value = '리드미컬'
    else:
        pred_value = '낭만-비애적인'
    test_pred_list.append(pred_value)
test_pred = np.array(test_pred_list)



"""

train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=0.1)

#NaN data처리 방법(3)
scaler = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0).fit(train_data.astype(float))
train_data = scaler.transform(train_data.astype(float))
test_data = scaler.transform(test_data.astype(float))


sub_train_data, sub_test_data, sub_train_answer, sub_test_answer = train_test_split(sub_data, sub_answer, test_size=0.1)
scaler = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0).fit(sub_train_data.astype(float))
sub_train_data = scaler.transform(sub_train_data.astype(float))
sub_test_data = scaler.transform(sub_test_data.astype(float))

"""
#Naive (0.76 0.78 0.83 0.85 0.88 0.90 0.92 1)
gnb = GaussianNB()
train_model = gnb.fit(train_data, train_answer)
#test_pred = train_model.predict(test_data)

"""
#KNN (0.59 0.64 0.66 0.71 0.74 0.76)
nbrs = KNeighborsClassifier(n_neighbors=7)
train_model = nbrs.fit(train_data, train_answer)
test_pred = train_model.predict(test_data)
before_pred = np.copy(test_pred)

print("바꾸기 전")
print(test_pred)

j = 0
for i in test_pred:
    if i == '리드미컬':
        j+=1
        continue
    else:
        gnb = GaussianNB()
        #train_model = gnb.fit(train_data, train_answer)              #61, 66(2), 71, 76(3), 80(2), 85,
        train_model = gnb.fit(sub_train_data, sub_train_answer)       #61, 71(2), 76(2), 85(3), 90, 95
        temp_data = test_data[j]
        temp_data = temp_data.reshape(1, -1)
        temp = train_model.predict(temp_data)
        test_pred[j] = temp[0]
        #print(test_pred[j])
        j+=1

print("바꾼 후")
print(test_pred)

before_correct_count = (before_pred == test_answer).sum()
accuracy2 = before_correct_count / len(test_answer)
correct_count = (test_pred == test_answer).sum()
accuracy1 = correct_count / len(test_answer)

print("원래 답")
print(test_answer)
print("바꾸기 전 Accuracy = " + str(accuracy2))
print("바꾼 후 Accuracy = " + str(accuracy1))


