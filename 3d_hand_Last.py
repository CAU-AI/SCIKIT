import numpy as np
import pandas as pd

from fastdtw import fastdtw
from sklearn.model_selection import train_test_split

import xlwt


def get_char(index):
    return chr((ord('a') + index))

def get_value(values):
    val = np.array(values.values)
    max_row = val.shape[0]
    max_col = val.shape[1]
    new_arr = [0] * max_row

    count_col = 0
    for j in range(0, max_row):
        count_row = 0
        for i in val[j]:
            if np.isnan(i):
                break
            count_row = count_row + 1
        new_data = [0] * count_row
        for i in range(0, count_row):
            new_data[i] = val[j][i]
            #new_arr[count_col].append(val[j][i])

        # new_arr[count_col] = get_bspline_arr(new_data)
        new_arr[count_col] = new_data
        count_col = count_col + 1
    new_arr = np.array(new_arr)
    return new_arr


print("start")
#xl = pd.ExcelFile("3D_handwriting_train.xlsx")
xl = [[]] * 26
for i in range (0, 26):
    xl[i] = pd.ExcelFile("train/" + chr((ord('a') + i)) + ".xlsx")

data_x = [[]] * 26
data_y = [[]] * 26
data_z = [[]] * 26
for i in range (0, 26):
    data_x[i] = pd.read_excel(io=xl[i], sheet_name=0, header=None)
    data_y[i] = pd.read_excel(io=xl[i], sheet_name=1, header=None)
    data_z[i] = pd.read_excel(io=xl[i], sheet_name=2, header=None)

X = [[]] * 26
Y = [[]] * 26
Z = [[]] * 26
for i in range (0, 26):
    X[i] = get_value(data_x[i])
    Y[i] = get_value(data_y[i])
    Z[i] = get_value(data_z[i])

print("end")
print("start test")


trainX = [[]] * 26
trainY = [[]] * 26
trainZ = [[]] * 26

testX = []
testY = []
testZ = []
testA = []

for i in range(0, 26):
    _trainX, _testX, _trainY, _testY, _trainZ, _testZ = train_test_split(X[i], Y[i], Z[i], test_size=0.1)
    len_train = np.array(_trainX).shape[0]
    len_test = np.array(_testX).shape[0]
    for j in range(0, len_train):
        trainX[i].append(_trainX[j])
        trainY[i].append(_trainY[j])
        trainZ[i].append(_trainZ[j])
    for j in range(0, len_test):
        testX.append(_testX[j])
        testY.append(_testY[j])
        testZ.append(_testZ[j])
        testA.append(get_char(i))

test_X = testX
test_Y = testY
test_Z = testZ

col_count = np.array(testA).shape[0]


print("end test")


def dtw(X, Y, Z, tX, tY, tZ):
    dist_x, path_x = fastdtw(X, tX)
    dist_y, path_y = fastdtw(Y, tY)
    dist_z, path_z = fastdtw(Z, tZ)
    dist = dist_x + dist_y + dist_z
    return dist

def fdsa(test_index):
    arr_min = [9999999999999] * 26
    arr_live = [0] * 26
    arr_count = [0] * 26
    for i in range(0, 25):
        for j in range(0, 26):
            if arr_live[j] == 0:
                loc_range = 0
                if i < 5:
                    loc_range = 2;
                elif 5 <= i < 10:
                    loc_range = 1
                else :
                    loc_range = 1

                for k in range(arr_count[j], arr_count[j] + loc_range):
                    dd = dtw(X[j][k], Y[j][k], Z[j][k], test_X[test_index], test_Y[test_index], test_Z[test_index])
                    if dd < arr_min[j]:
                        arr_min[j] = dd
                        #print("find : " + str(j) + "\tmin : " + str(arr_min[j]) + "\tdd : " + str(dd))

                arr_count[j] = arr_count[j] + loc_range

        max_index = 0
        max_value = 0
        for j in range(0, 26):
            #print("dd : " + str(j) + ",\tvalue : " + str(arr_min[j]))
            if arr_live[j] == 0:
                if max_value < arr_min[j]:
                    max_index = j
                    max_value = arr_min[j]

        arr_live[max_index] = 1
        #print("maxIndex : " + str(max_index) + ",\tvalue : " + str(max_value))

    ret = 0
    value = 0
    for i in range(0, 26):
        if arr_live[i] == 0:
            ret = i
            value = arr_min[i]
    return ret, value


c = 0
for i in range(0, 130):
    dd, value = fdsa(i)
    ss = testA[i][0]
    val = get_char(dd)
    print("correct : " + ss + "\tfind : " + val + "\tvalue : " + str(value))
    if ss == val:
        c += 1

acc = (c / 130) * 100
print("Acc : " + str(acc) + " %")




def make(save_name, tag, X, Y, Z):
    col_count = np.array(X).shape[0]
    score = [0] * col_count
    score_b = [0] * col_count

    for i in range(0, col_count):
        sum = 0
        for j in range(0, col_count):
            if i != j:
                dist_x, path_x = fastdtw(X[i], X[j])
                dist_y, path_y = fastdtw(Y[i], Y[j])
                dist_z, path_z = fastdtw(Z[i], Z[j])
                dist = dist_x + dist_y + dist_z
                sum = dist * dist
        score[i] = sum
        print(tag + "\tsum : " + str(score[i]))

    workbook = xlwt.Workbook(encoding='utf-8')  # utf-8 인코딩 방식의 workbook 생성
    ws_x = workbook.add_sheet("x")  # 시트 생성
    ws_y = workbook.add_sheet("y")  # 시트 생성
    ws_z = workbook.add_sheet("z")  # 시트 생성

    for i in range(0, col_count):
        min = 9999999999999
        min_index = 0;
        for j in range(0, col_count):
            if score_b[j] == 0:
                if score[j] < min:
                    min = score[j]
                    min_index = j
        score_b[min_index] = 1

        print("result min : " + str(min) + "\tindex : " + str(min_index))
        xlen = np.array(X[min_index]).shape[0]
        for j in range(0, xlen):
            ws_x.write(i, j, X[min_index][j])
            ws_y.write(i, j, Y[min_index][j])
            ws_z.write(i, j, Z[min_index][j])

    workbook.save(save_name)








