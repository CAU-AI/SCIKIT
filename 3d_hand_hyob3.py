import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import xlwt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import BSpline

from scipy.spatial.distance import euclidean

from fastdtw import fastdtw



def get_alphabet(index):
    if index == 0:  return 'a'
    elif index == 1:  return 'b'
    elif index == 2:  return 'c'
    elif index == 3:  return 'd'
    elif index == 4:  return 'e'
    elif index == 5:  return 'f'
    elif index == 6:  return 'g'
    elif index == 7:  return 'h'
    elif index == 8:  return 'i'
    elif index == 9:  return 'j'
    elif index == 10:  return 'k'
    elif index == 11:  return 'l'
    elif index == 12:  return 'm'
    elif index == 13:  return 'n'
    elif index == 14:  return 'o'
    elif index == 15:  return 'p'
    elif index == 16:  return 'q'
    elif index == 17:  return 'r'
    elif index == 18:  return 's'
    elif index == 19:  return 't'
    elif index == 20:  return 'u'
    elif index == 21:  return 'v'
    elif index == 22:  return 'w'
    elif index == 23:  return 'x'
    elif index == 24:  return 'y'
    else: return 'z'


def get_eval(x, y, z):
    size = np.array(x).shape[0]
    sum_x = 0
    sum_y = 0
    sum_z = 0
    for i in range(0, size):
        sum_x += x
        sum_y += y
        sum_z += z
    eval_x = sum_x/size
    eval_y = sum_y/size
    eval_z = sum_z/size
    return eval_x, eval_y, eval_z


def get_pca(x, y, z):
    feature_size = 2;
    pca = PCA(n_components=feature_size)
    data = []

    size = np.array(x).shape[0]
    for i in range(0, size):
        dd = []
        dd.append(x[i])
        dd.append(y[i])
        dd.append(z[i])
        data.append(dd)
    pca.fit(data)
    data = pca.transform(data)
    data = data.T
    return data


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


print("start")
train_xl = pd.ExcelFile("3D_handwriting_train_data_train.xls")
test_xl = pd.ExcelFile("3D_handwriting_train_data_test.xlsx")
print("end")


alphabet = []
for i in range(0, 26):
    alphabet.append(get_value(pd.read_excel(io=train_xl, sheet_name=i, header=None)))


data_x = pd.read_excel(io=test_xl, sheet_name=0, header=None)
data_y = pd.read_excel(io=test_xl, sheet_name=1, header=None)
data_z = pd.read_excel(io=test_xl, sheet_name=2, header=None)
data_a = pd.read_excel(io=test_xl, sheet_name=3, header=None)

X = get_value(data_x)
Y = get_value(data_y)
Z = get_value(data_z)
A = data_a.values



def test(alpha, test_x, test_y, test_z, test_a):
    acc = 0

    bool_alphabet = [0] * 26
    sum_alphabet = [0] * 26

    for i in range(0, 25):
        max_index = 0
        max_sum = 0
        for j in range(0, 26):
            if(bool_alphabet[j] == 0):
                ping_pong = 6
                dist = dtw_distance(alpha[j][i * ping_pong + 0], alpha[j][i * ping_pong + 1], alpha[j][i * ping_pong + 2], test_x, test_y, test_z)
                dist += dtw_distance(alpha[j][i * ping_pong + 3], alpha[j][i * ping_pong + 4], alpha[j][i * ping_pong + 5], test_x, test_y, test_z)
                #dist += dtw_distance(alpha[j][i * 9 + 6], alpha[j][i * 9 + 7], alpha[j][i * 9 + 8], test_x, test_y, test_z)
                dist = dist/(ping_pong/3)
                sum_alphabet[j] += dist
                if max_sum < sum_alphabet[j]:
                    max_sum = sum_alphabet[j]
                    max_index = j
        bool_alphabet[max_index] = 1
        print("test_ans : " + test_a + ",\tdel_alphabet : " + get_alphabet(max_index))

    last_index = 0

    for i in range(0, 26):
        if bool_alphabet[i] == 0:
            last_index = i
    if get_alphabet(last_index) == test_a:
        print("correct num : " + test_a)
        return 1
    else:
        print("wrong")
        return 0

acc = 0

for i in range(0, np.array(X).shape[0]):
    acc += test(alphabet, X[i], Y[i], Z[i], A[i])

print("Acc : " + str(100 * acc/np.array(X).shape[0]) + " %")

