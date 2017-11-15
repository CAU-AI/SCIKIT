import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import BSpline

from scipy.spatial.distance import euclidean

from fastdtw import fastdtw


def get_bspline_arr(arrs, len = 50):
    arrs_len = np.array(arrs).shape[0]
    t = []
    for i in range(0, arrs_len):
        t.append(i)
    spl = BSpline(t, arrs, 2)
    spl(2.5)
    ret = [0] * len

    for i in range(0, len):
        ret[i] = spl(i * arrs_len / len)
    return ret

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

def d(s, t):
    ff = s - t;
    if ff < 0:
        ff = ff * -1
    return ff

def minimum(a, b, c):
    if a < b:
        if c < a:
            return c
        else:
            return a
    else:
        if c < b:
            return c
        else:
            return b

def dtw_distance(tx, ty, tz, x, y, z):
    arr_tx = [0]
    arr_ty = [0]
    arr_tz = [0]
    arr_x = [0]
    arr_y = [0]
    arr_z = [0]
    len_t = np.array(tx).shape[0]
    len_x = np.array(x).shape[0]
    for i in range(0, len_t):
        arr_tx.append(tx[i])
        arr_ty.append(ty[i])
        arr_tz.append(tz[i])
    for i in range(0, len_x):
        arr_x.append(x[i])
        arr_y.append(y[i])
        arr_z.append(z[i])

    len_t = np.array(arr_tx).shape[0]
    len_x = np.array(arr_x).shape[0]

    dtw = [[0] * len_x] * len_t

    #print(arr_tx)

    for i in range(1, len_t):
       dtw[i][0] = 9999999
    for i in range(1, len_x):
       dtw[0][i] = 9999999
    dtw[0][0] = 0

    for i in range(1, len_t):
        for j in range(1, len_x):
            cost = d(arr_tx[i], arr_x[j]) + d(arr_ty[i], arr_y[j]) + d(arr_tz[i], arr_z[j])
            dtw[i][j] = cost + minimum(dtw[i-1][j],    #insertion
                                       dtw[i][j-1],    #deletion
                                       dtw[i-1][j-1])    #match
    return dtw[len_t - 1][len_x - 1]

def dtw_distance(tx, ty, tz, x, y, z):
    arr_tx = [0]
    arr_ty = [0]
    arr_tz = [0]
    arr_x = [0]
    arr_y = [0]
    arr_z = [0]
    len_t = np.array(tx).shape[0]
    len_x = np.array(x).shape[0]
    for i in range(0, len_t):
        arr_tx.append(tx[i])
        arr_ty.append(ty[i])
        arr_tz.append(tz[i])
    for i in range(0, len_x):
        arr_x.append(x[i])
        arr_y.append(y[i])
        arr_z.append(z[i])

    len_t = np.array(arr_tx).shape[0]
    len_x = np.array(arr_x).shape[0]

    dtw = [[0] * len_x] * len_t

    #print(arr_tx)

    for i in range(1, len_t):
       dtw[i][0] = 9999999
    for i in range(1, len_x):
       dtw[0][i] = 9999999
    dtw[0][0] = 0

    for i in range(1, len_t):
        for j in range(1, len_x):
            cost = d(arr_tx[i], arr_x[j]) + d(arr_ty[i], arr_y[j]) + d(arr_tz[i], arr_z[j])
            dtw[i][j] = cost + minimum(dtw[i-1][j],    #insertion
                                       dtw[i][j-1],    #deletion
                                       dtw[i-1][j-1])    #match
    return dtw[len_t - 1][len_x - 1]

print("start")
xl = pd.ExcelFile("3D_handwriting_train.xlsx")
print("end")


data_x = pd.read_excel(io=xl, sheet_name=0, header=None)
data_y = pd.read_excel(io=xl, sheet_name=1, header=None)
data_z = pd.read_excel(io=xl, sheet_name=2, header=None)
data_a = pd.read_excel(io=xl, sheet_name=3, header=None)


X = get_value(data_x)
Y = get_value(data_y)
Z = get_value(data_z)
A = data_a.values

train_x, test_x, train_y, test_y, train_z, test_z, train_a, test_a = train_test_split(X, Y, Z, A, test_size=0.05)


len_train = np.array(train_x).shape[0]
len_test = np.array(test_x).shape[0]

correct = 0
for i in range(0, len_test):
    loc_min = 999999999
    loc_min_answer = [0]
    for j in range(0, len_train):
        dist_x, path_x = fastdtw(train_x[j], test_x[i], dist=euclidean)
        dist_y, path_y = fastdtw(train_y[j], test_y[i], dist=euclidean)
        dist_z, path_z = fastdtw(train_z[j], test_z[i], dist=euclidean)
        dist = dist_x + dist_y + dist_z
        #dist = dtw_distance(train_x[j], train_y[j], train_z[j], test_x[i], test_y[i], test_z[i])
        if dist < loc_min:
            loc_min = dist
            loc_min_answer = np.array(train_a[j])[0]
        print("percentage : " + str(100 * j/len_train) + " ,\ttrain : " + loc_min_answer + " ,\ttest : " + np.array(test_a[i])[0])
    print("train : " + loc_min_answer + " , test : " + np.array(test_a[i])[0])
    if loc_min_answer == np.array(test_a[i])[0]:
        correct = correct + 1

print("Acc : ")
print(100 * correct/len_test)




