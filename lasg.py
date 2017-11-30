from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import feature_selection as aaa

xl = pd.ExcelFile("Music_style_train.xlsx")
data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)
data = np.array(data_excel.values)
answer = np.array(answer_excel.values).flatten().transpose()


train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=0.2)

test_pred_list = []
list1 = [90, 22, 21, 46, 36, 84, 66, 81, 3, 8]
list2 = [360, 364, 342, 354, 47, 53, 25, 24, 21, 22]
list3 = [257, 258, 347, 250, 254, 256, 259, 261, 204, 253]

list = list1+list2+list3

for i in range(len(test_data)):
    for j in range(len(train_data)):
        a=0
        distance = []

        for k in range(len(test_data[0])):
            if k in list:
                continue
            else:
                 a += (test_data[i][k] - train_data[j][k]) ** 2
            b= a**0.5
            distance.append(b)

        label1 = 0
        label2 = 0
        label3 = 0

        origin_distance = distance[:]
        distance.sort()

    for p in range(0, 4):
        index = origin_distance.index(distance[p])
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
        #print(len(test_pred))
        #print(len(test_answer))


print(len(test_pred))
print(len(test_answer))
correct_count = (test_pred == test_answer).sum()
#print(correct_count, len(test_answer))

accuracy = correct_count / len(test_answer)
print("Accuracy = " + str(accuracy))






















'''
test_pred_list=[]
for i in range(len(test_data)):
    a=0
    b=0
    c=0
    distance_range = []
    origin_distance_range=[]
    E1 = aaa.e(0, 69)
    E2 = aaa.e(69, 140)
    E3 = aaa.e(140, 209)

    for j in range(len(test_data[0])):

        a += (test_data[i][j] - E1[j]) ** 2
        b += (test_data[i][j] - E2[j]) ** 2
        c += (test_data[i][j] - E3[j]) ** 2

    d = min(a,b,c)
    if min(a, b, c) == a:
        pred_value = '멜랑꼴리'
    elif min(a, b, c) == b:
        pred_value = '리드미컬'
    else:
        pred_value = '낭만-비애적인'
    test_pred_list.append(pred_value)
test_pred = np.array(test_pred_list)

correct_count = (test_pred == test_answer).sum()
print(correct_count, len(test_answer))
accuracy = correct_count / len(test_answer)
print("Accuracy = " + str(accuracy))

'''