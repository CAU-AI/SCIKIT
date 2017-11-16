import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import BSpline


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

def dtw_distance(tx, ty, x, y):
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
    return dtw[len_t - 1][len_x - 1]


print("start_3d")
xl = pd.ExcelFile("3D_handwriting_train.xlsx")
print("end_3d")


data_x = pd.read_excel(io=xl, sheet_name=0, header=None)
data_y = pd.read_excel(io=xl, sheet_name=1, header=None)
data_z = pd.read_excel(io=xl, sheet_name=2, header=None)
data_a = pd.read_excel(io=xl, sheet_name=3, header=None)


X = get_value(data_x)
Y = get_value(data_y)
Z = get_value(data_z)
A = data_a.values


al_a = []
al_b = []
al_c = []
al_d = []
al_e = []
al_f = []
al_g = []
al_h = []
al_i = []
al_j = []
al_k = []
al_l = []
al_m = []
al_n = []
al_o = []
al_p = []
al_q = []
al_r = []
al_s = []
al_t = []
al_u = []
al_v = []
al_w = []
al_x = []
al_y = []
al_z = []

size = np.array(A).shape[0]
for i in range(0, size):
    if A[i][0] == 'a':
        al_a.append(X[i])
        al_a.append(Y[i])
        al_a.append(Z[i])
    elif A[i][0] == 'b':
        al_b.append(X[i])
        al_b.append(Y[i])
        al_b.append(Z[i])
    elif A[i][0] == 'c':
        al_c.append(X[i])
        al_c.append(Y[i])
        al_c.append(Z[i])
    elif A[i][0] == 'd':
        al_d.append(X[i])
        al_d.append(Y[i])
        al_d.append(Z[i])
    elif A[i][0] == 'e':
        al_e.append(X[i])
        al_e.append(Y[i])
        al_e.append(Z[i])
    elif A[i][0] == 'f':
        al_f.append(X[i])
        al_f.append(Y[i])
        al_f.append(Z[i])
    elif A[i][0] == 'g':
        al_g.append(X[i])
        al_g.append(Y[i])
        al_g.append(Z[i])
    elif A[i][0] == 'h':
        al_h.append(X[i])
        al_h.append(Y[i])
        al_h.append(Z[i])
    elif A[i][0] == 'i':
        al_i.append(X[i])
        al_i.append(Y[i])
        al_i.append(Z[i])
    elif A[i][0] == 'j':
        al_j.append(X[i])
        al_j.append(Y[i])
        al_j.append(Z[i])
    elif A[i][0] == 'k':
        al_k.append(X[i])
        al_k.append(Y[i])
        al_k.append(Z[i])
    elif A[i][0] == 'l':
        al_l.append(X[i])
        al_l.append(Y[i])
        al_l.append(Z[i])
    elif A[i][0] == 'm':
        al_m.append(X[i])
        al_m.append(Y[i])
        al_m.append(Z[i])
    elif A[i][0] == 'n':
        al_n.append(X[i])
        al_n.append(Y[i])
        al_n.append(Z[i])
    elif A[i][0] == 'o':
        al_o.append(X[i])
        al_o.append(Y[i])
        al_o.append(Z[i])
    elif A[i][0] == 'p':
        al_p.append(X[i])
        al_p.append(Y[i])
        al_p.append(Z[i])
    elif A[i][0] == 'q':
        al_q.append(X[i])
        al_q.append(Y[i])
        al_q.append(Z[i])
    elif A[i][0] == 'r':
        al_r.append(X[i])
        al_r.append(Y[i])
        al_r.append(Z[i])
    elif A[i][0] == 's':
        al_s.append(X[i])
        al_s.append(Y[i])
        al_s.append(Z[i])
    elif A[i][0] == 't':
        al_t.append(X[i])
        al_t.append(Y[i])
        al_t.append(Z[i])
    elif A[i][0] == 'u':
        al_u.append(X[i])
        al_u.append(Y[i])
        al_u.append(Z[i])
    elif A[i][0] == 'v':
        al_v.append(X[i])
        al_v.append(Y[i])
        al_v.append(Z[i])
    elif A[i][0] == 'w':
        al_w.append(X[i])
        al_w.append(Y[i])
        al_w.append(Z[i])
    elif A[i][0] == 'x':
        al_x.append(X[i])
        al_x.append(Y[i])
        al_x.append(Z[i])
    elif A[i][0] == 'y':
        al_y.append(X[i])
        al_y.append(Y[i])
        al_y.append(Z[i])
    elif A[i][0] == 'z':
        al_z.append(X[i])
        al_z.append(Y[i])
        al_z.append(Z[i])

print("start get_arr_pca")
al_a = get_arr_pca(al_a)
al_b = get_arr_pca(al_b)
al_c = get_arr_pca(al_c)
al_d = get_arr_pca(al_d)
al_e = get_arr_pca(al_e)
al_f = get_arr_pca(al_f)
al_g = get_arr_pca(al_g)
al_h = get_arr_pca(al_h)
al_i = get_arr_pca(al_i)
al_j = get_arr_pca(al_j)
al_k = get_arr_pca(al_k)
al_l = get_arr_pca(al_l)
al_m = get_arr_pca(al_m)
al_n = get_arr_pca(al_n)
al_o = get_arr_pca(al_o)
al_p = get_arr_pca(al_p)
al_q = get_arr_pca(al_q)
al_r = get_arr_pca(al_r)
al_s = get_arr_pca(al_s)
al_t = get_arr_pca(al_t)
al_u = get_arr_pca(al_u)
al_v = get_arr_pca(al_v)
al_w = get_arr_pca(al_w)
al_x = get_arr_pca(al_x)
al_y = get_arr_pca(al_y)
al_z = get_arr_pca(al_z)
print("end get_arr_pca")

print("start")
xl = pd.ExcelFile("3D_handwriting_train_data.xls")
print("end")

alphabet = []
for i in range(0, 26):
    alphabet.append(get_value(pd.read_excel(io=xl, sheet_name=i, header=None)))

test_num = 9

def test(al, num, test_num):
    acc = 0

    max_size = np.array(al).shape[0]
    for als in range(0, test_num):
        test_x = al[max_size - 30 + als * 2]
        test_y = al[max_size - 30 + als * 2 + 1]

        bool_alphabet = [0] * 26
        sum_alphabet = [0] * 26

        for i in range(0, 25):
            max_index = 0
            max_sum = 0
            for j in range(0, 26):
                if(bool_alphabet[j] == 0):
                    dist = dtw_distance(alphabet[j][i * 2], alphabet[j][i * 2 + 1], test_x, test_y)
                    sum_alphabet[j] += dist
                    if sum_alphabet[j] > max_sum:
                        max_sum = sum_alphabet[j]
                        max_index = j
            bool_alphabet[max_index] = 1
            #print("num : " + str(num) + ",\tindex : " + str(max_index))

        last_index = 0

        for i in range(0, 26):
            if bool_alphabet[i] == 0:
                last_index = i
        if last_index == num:
            acc += 1
            print("correct num : " + str(num))
        else:
            print("wrong")
    return acc

acc = 0
acc += test(al_a, 0, test_num)
acc += test(al_b, 1, test_num)
acc += test(al_c, 2, test_num)
acc += test(al_d, 3, test_num)
acc += test(al_e, 4, test_num)
acc += test(al_f, 5, test_num)
acc += test(al_g, 6, test_num)
acc += test(al_h, 7, test_num)
acc += test(al_i, 8, test_num)
acc += test(al_j, 9, test_num)
acc += test(al_k, 10, test_num)
acc += test(al_l, 11, test_num)
acc += test(al_m, 12, test_num)
acc += test(al_n, 13, test_num)
acc += test(al_o, 14, test_num)
acc += test(al_p, 15, test_num)
acc += test(al_q, 16, test_num)
acc += test(al_r, 17, test_num)
acc += test(al_s, 18, test_num)
acc += test(al_t, 19, test_num)
acc += test(al_u, 20, test_num)
acc += test(al_v, 21, test_num)
acc += test(al_w, 22, test_num)
acc += test(al_x, 23, test_num)
acc += test(al_y, 24, test_num)
acc += test(al_z, 25, test_num)
print("Acc : " + str(100 * acc/(test_num * 26)) + " %")
'''

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
    for j in range(0, len_train - 1):
        dist = dtw_distance(train_x[j], train_y[j], train_z[j], test_x[i], test_y[i], test_z[i])
        if dist < loc_min:
            loc_min = dist
            loc_min_answer = np.array(train_a[j])[0]
    print("train : " + loc_min_answer + " , test : " + np.array(test_a[i])[0])
    if loc_min_answer == np.array(test_a[i])[0]:
        correct = correct + 1

print("Acc : ")
print(100 * correct/len_test)
'''




