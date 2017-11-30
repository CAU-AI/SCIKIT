from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

xl = pd.ExcelFile("Music_style_train.xlsx")
x2 = pd.ExcelFile("Music_Demo.xlsx")
data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)
demo_excel = pd.read_excel(io=x2, sheetName = 0, header = None)

data = np.array(data_excel.values)                                  #210 * 376
answer = np.array(answer_excel.values).flatten().transpose()
demo = np.array(demo_excel.values)
demo_answer = []

new_data = np.zeros((210,376),dtype="i")

for i in range(len(data)):
    for j in range(len(data[0])):
        if np.isnan(data[i,j]):                          #NaN인지 확인
            new_data[i, j] = 0
        else:
            new_data[i, j] = 1

for i in range(0, len(demo)):      #test_data 크기
    distance_range = []
    for j in range(len(data)):
        I_value = 0
        a = 0.0
        for k in range(len(data[0])):
            I_value += new_data[j][k]
            if np.isnan(data[j][k]):
                difference = 0.0
            else:
                difference = abs(data[j][k] - data[i][k])
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
        if answer[index] == '멜랑꼴리':
            label1 += 1
        elif answer[index] == '리드미컬':
            label2 += 1
        else:
            label3 += 1
    if max(label1, label2, label3) == label1:
        pred_value = '멜랑꼴리'
    elif max(label1, label2, label3) == label2:
        pred_value = '리드미컬'
    else:
        pred_value = '낭만-비애적인'

    demo_answer.append(pred_value)


print(demo_answer)
print(len(demo_answer))






# Nan median
# Naive bayes
#partial_distance strategy

