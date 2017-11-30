import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import BSpline


xl = pd.ExcelFile("3D_handwriting_train.xlsx")
demo_xl = pd.ExcelFile("3D_handwriting_train_data_test.xlsx")
f = open('result3.txt', 'w+t')

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

def get_arr_pca(al):
    size = int(np.array(al).shape[0]/3)
    ret = []
    for i in range(0, size):
        buf = get_pca(al[3 * i], al[3 * i + 1], al[3 * i + 2])
        ret.append(buf[0])
        ret.append(buf[1])
    return ret

def get_bspline_arr(arrs, len = 10):
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

def dtw_distance(tx, ty, tz, x, y, z, max):
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
            if i == j:
                if dtw[i][i] > max:
                    return max + 1
    return dtw[len_t - 1][len_x - 1]

def dtw_distance2(tx, ty, x, y, max):
    arr_tx = [0]
    arr_ty = [0]
    arr_x = [0]
    arr_y = [0]
    len_t = np.array(tx).shape[0]
    len_x = np.array(x).shape[0]
    for i in range(0, len_t):
        arr_tx.append(tx[i])
        arr_ty.append(ty[i])
    for i in range(0, len_x):
        arr_x.append(x[i])
        arr_y.append(y[i])

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
            cost = d(arr_tx[i], arr_x[j]) + d(arr_ty[i], arr_y[j])
            dtw[i][j] = cost + minimum(dtw[i-1][j],    #insertion
                                       dtw[i][j-1],    #deletion
                                       dtw[i-1][j-1])    #match
            if i == j:
                if dtw[i][i] > max:
                    return max + 1
    return dtw[len_t - 1][len_x - 1]

def dtw_distance1(tx, x, max):
    arr_tx = [0]
    arr_x = [0]
    len_t = np.array(tx).shape[0]
    len_x = np.array(x).shape[0]
    for i in range(0, len_t):
        arr_tx.append(tx[i])
    for i in range(0, len_x):
        arr_x.append(x[i])

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
            cost = d(arr_tx[i], arr_x[j])
            dtw[i][j] = cost + minimum(dtw[i-1][j],    #insertion
                                       dtw[i][j-1],    #deletion
                                       dtw[i-1][j-1])    #match
            if i == j:
                if dtw[i][i] > max:
                    return max + 1
    return dtw[len_t - 1][len_x - 1]


print("start")

data_x = pd.read_excel(io=xl, sheetname=0, header=None)
data_y = pd.read_excel(io=xl, sheetname=1, header=None)
data_z = pd.read_excel(io=xl, sheetname=2, header=None)
data_a = pd.read_excel(io=xl, sheetname=3, header=None)

X = get_value(data_x)
Y = get_value(data_y)
Z = get_value(data_z)
A = data_a.values

for i in range(0, np.array(X).shape[0]):
    X[i] = get_bspline_arr(X[i])
    Y[i] = get_bspline_arr(Y[i])
    Z[i] = get_bspline_arr(Z[i])

demo_ex_x = pd.read_excel(io=demo_xl, sheetname=0, header=None)
demo_ex_y = pd.read_excel(io=demo_xl, sheetname=1, header=None)
demo_ex_z = pd.read_excel(io=demo_xl, sheetname=2, header=None)
demo_ex_a = pd.read_excel(io=demo_xl, sheetname=3, header=None)


train_x, test_x, train_y, test_y, train_z, test_z, train_a, test_a = train_test_split(X, Y, Z, A, test_size=0.01)
test_x = get_value(demo_ex_x)
test_y = get_value(demo_ex_y)
test_z = get_value(demo_ex_z)
test_a = demo_ex_a.values


len_train = np.array(train_x).shape[0]
len_test = np.array(test_x).shape[0]

correct = 0
test_case = 0
for i in range(0, 1):
    loc_min = 20
    loc_min_answer = [0]
    for j in range(0, len_train - 1):
        dist = dtw_distance(train_x[j], train_y[j], train_z[j], test_x[i], test_y[i], test_z[i], loc_min)
        # dist = dtw_distance2(train_x[j], train_y[j], test_x[i], test_y[i], loc_min)
        if dist < loc_min:
            loc_min = dist
            loc_min_answer = np.array(train_a[j])[0]
    print("train : " + loc_min_answer + " , test : " + np.array(test_a[i])[0] + ", dist : " + str(loc_min))
    if loc_min_answer == np.array(test_a[i])[0]:
        correct = correct + 1



