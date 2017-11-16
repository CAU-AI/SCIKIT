from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

xl = pd.ExcelFile("Music_style_train.xlsx")
data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)
data = np.array(data_excel.values)                                  #210 * 376
answer = np.array(answer_excel.values).flatten().transpose()

slice=[]   #data에서 NaN이 존재하는 열 index 뽑아내기  -> nan이 존재하는 열은 계산 대상에 포함X
for i in range(len(data.T)):                    #data.T = 376 * 210
    for j in range(len(data.T[0])):
        if np.isnan(data.T[i][j]):
            slice.append(i)
            break
#print(slice)
#print(len(data))
#print(len(data[0]))
accuracy_list=[]

def predict(num):
    set_test_size = 0.2
    compare_num = num    # sort한 list에서 label수 비교할 num
    train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=set_test_size)

    new_data = np.zeros((210, 376), dtype="i")
#test_pred = np.array([],dtype="str")
    test_pred_list =[]

    I_value = len(test_data[0]) - len(slice)
    for i in range(0, len(test_data)):      #test_data 크기
        distance_range = []
        for j in range(len(train_data)):
            a = 0.0
            for k in range(len(train_data[0])):
                if k in slice:
                    continue
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

    correct_count = (test_pred == test_answer).sum()
    print("Correct = " + str(correct_count),  " test_length = " + str(len(test_answer)))
    accuracy = correct_count / len(test_answer)
    print("K:" + str(num) + " Accuracy = " + str(accuracy))
    accuracy_list.append(accuracy)

range_num = 10
for i in range(1,range_num):                   # 이웃 수 변화
    predict(i)
print("Max_Accuracy = " + str(max(accuracy_list)))

x=range(1,range_num )
plt.plot(x, accuracy_list, 'ro')
plt.show()

