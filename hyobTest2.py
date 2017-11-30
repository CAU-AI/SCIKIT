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

print("start")
xl = pd.ExcelFile("3D_handwriting_train_data_train.xlsx")
print("end")


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

'''
def get_arr_pca(al):
    size = int(np.array(al).shape[0]/3)
    ret = []
    for i in range(0, size):
        buf = get_pca(al[3 * i], al[3 * i + 1], al[3 * i + 2])
        ret.append(buf[0])
        ret.append(buf[1])
    return ret

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

'''


def get_sort_arr(al):
    size = int(np.array(al).shape[0]/2)

    rank = []
    rank_bool = []
    ret = []
    for i in range(0, size):
        sum = 0
        for j in range(0, size):
            dist_x, path_x = fastdtw(al[2 * i + 0], al[2 * j + 0])
            dist_y, path_y = fastdtw(al[2 * i + 1], al[2 * j + 1])
            sum += dist_y + dist_x
        print("sum : " + str(sum))
        rank.append(sum)
        rank_bool.append(0)
    for i in range(0, size):
        min = 9999999999999
        min_index = 0
        for j in range(0, size):
            if rank_bool[j] == 0:
                if rank[j] < min:
                    min = rank[j]
                    min_index = j
        rank_bool[min_index] = 1
        print("ranking : " + str(rank[min_index]))
        ret.append(al[2 * min_index])
        ret.append(al[2 * min_index + 1])
    return ret

#al_a = get_sort_arr(al_a)

def get_worksheet(ws, al):
    for row in range(0, np.array(al).shape[0] - 30):
        for col in range(0, np.array(al[row]).shape[0]):
            ws.write(row, col, u"" + str(al[row][col]))

workbook = xlwt.Workbook(encoding='utf-8')
workbook.default_style.font.height = 20 * 11
worksheet = []
for i in range(0, 26):
    worksheet.append(workbook.add_sheet(u"al_" + str(i)))
get_worksheet(worksheet[0], al_a)
get_worksheet(worksheet[1], al_b)
get_worksheet(worksheet[2], al_c)
get_worksheet(worksheet[3], al_d)
get_worksheet(worksheet[4], al_e)
get_worksheet(worksheet[5], al_f)
get_worksheet(worksheet[6], al_g)
get_worksheet(worksheet[7], al_h)
get_worksheet(worksheet[8], al_i)
get_worksheet(worksheet[9], al_j)
get_worksheet(worksheet[10], al_k)
get_worksheet(worksheet[11], al_l)
get_worksheet(worksheet[12], al_m)
get_worksheet(worksheet[13], al_n)
get_worksheet(worksheet[14], al_o)
get_worksheet(worksheet[15], al_p)
get_worksheet(worksheet[16], al_q)
get_worksheet(worksheet[17], al_r)
get_worksheet(worksheet[18], al_s)
get_worksheet(worksheet[19], al_t)
get_worksheet(worksheet[20], al_u)
get_worksheet(worksheet[21], al_v)
get_worksheet(worksheet[22], al_w)
get_worksheet(worksheet[23], al_x)
get_worksheet(worksheet[24], al_y)
get_worksheet(worksheet[25], al_z)

workbook.save('3D_handwriting_train_data_train.xls')



'''
train_x, test_x, train_y, test_y, train_z, test_z, train_a, test_a = train_test_split(X, Y, Z, A, test_size=0.05)


len_train = np.array(train_x).shape[0]
len_test = np.array(test_x).shape[0]

correct = 0
for i in range(0, len_test):
    loc_min = 999999999
    loc_min_answer = [0]
    for j in range(0, len_train):
        dist_x, path_x = fastdtw(train_x[j], test_x[i])
        dist_y, path_y = fastdtw(train_y[j], test_y[i])
        #dist_z, path_z = fastdtw(train_z[j], test_z[i])
        dist = dist_x + dist_y
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

'''



