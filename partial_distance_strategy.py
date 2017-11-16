from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.decomposition import PCA

xl = pd.ExcelFile("Music_style_train.xlsx")
data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)
data = np.array(data_excel.values)                                  #210 * 376
answer = np.array(answer_excel.values).flatten().transpose()



# partial_distance_strategy 식 사용하여 거리 sort후 k 개의 data구분하여 가장 많은 label이 나온 값으로 label 지정

set_test_size = 0.2
compare_num = 4    # sort한 list에서 label수 비교할 num
train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=set_test_size)

new_data = np.zeros((210,376),dtype="i")

#test_pred = np.array([],dtype="str")
test_pred_list =[]

for i in range(len(data)):
    for j in range(len(data[0])):
        if np.isnan(data[i,j]):                          #NaN인지 확인
            new_data[i, j] = 0
        else:
            new_data[i, j] = 1

for i in range(0, int(len(data) * set_test_size)):      #test_data 크기
    distance_range = []
    for j in range(len(train_data)):
        I_value = 0    
        a = 0
        for k in range(len(train_data[0])):
            # test_data[i][k] , train_data[j][k]
            I_value += new_data[j][k]
            if np.isnan(train_data[j][k]) or np.isnan(test_data[i][k]):
                difference = 0
            else:
                difference = abs(train_data[j][k] - test_data[i][k])
            b = difference * new_data[j][k]
            a += b
        distance = len(data[0]) / I_value * a
        distance_range.append(distance)

    label1 = 0
    label2 = 0
    label3 = 0
    origin_distance_range = distance_range[:]
    distance_range.sort()                 # len(distance_range) = 168, len(train_data) = 168, len(test_data) = 42    test_size = 0.2

    for p in range(0, compare_num):
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

print(test_pred)
print(test_answer)

correct_count = (test_pred == test_answer).sum()
print(correct_count, len(test_answer))
accuracy = correct_count / len(test_answer)
print("Accuracy = " + str(accuracy))
