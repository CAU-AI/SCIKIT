import numpy as np
import pandas as pd

from fastdtw import fastdtw

import xlwt


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
xl_a = pd.ExcelFile("a.xlsx")
xl_b = pd.ExcelFile("b.xlsx")
xl_c = pd.ExcelFile("c.xlsx")
xl_d = pd.ExcelFile("d.xlsx")
xl_e = pd.ExcelFile("e.xlsx")
xl_f = pd.ExcelFile("f.xlsx")
xl_g = pd.ExcelFile("g.xlsx")
xl_h = pd.ExcelFile("h.xlsx")
xl_i = pd.ExcelFile("i.xlsx")
xl_j = pd.ExcelFile("j.xlsx")
xl_k = pd.ExcelFile("k.xlsx")
xl_l = pd.ExcelFile("l.xlsx")
xl_m = pd.ExcelFile("m.xlsx")
xl_n = pd.ExcelFile("n.xlsx")
xl_o = pd.ExcelFile("o.xlsx")
xl_p = pd.ExcelFile("p.xlsx")
xl_q = pd.ExcelFile("q.xlsx")
xl_r = pd.ExcelFile("r.xlsx")
xl_s = pd.ExcelFile("s.xlsx")
xl_t = pd.ExcelFile("t.xlsx")
xl_u = pd.ExcelFile("u.xlsx")
xl_v = pd.ExcelFile("v.xlsx")
xl_w = pd.ExcelFile("w.xlsx")
xl_x = pd.ExcelFile("x.xlsx")
xl_y = pd.ExcelFile("y.xlsx")
xl_z = pd.ExcelFile("z.xlsx")
print("end")

data_a_x = pd.read_excel(io=xl_a, sheet_name=0, header=None)
data_b_x = pd.read_excel(io=xl_b, sheet_name=0, header=None)
data_c_x = pd.read_excel(io=xl_c, sheet_name=0, header=None)
data_d_x = pd.read_excel(io=xl_d, sheet_name=0, header=None)
data_e_x = pd.read_excel(io=xl_e, sheet_name=0, header=None)
data_f_x = pd.read_excel(io=xl_f, sheet_name=0, header=None)
data_g_x = pd.read_excel(io=xl_g, sheet_name=0, header=None)
data_h_x = pd.read_excel(io=xl_h, sheet_name=0, header=None)
data_i_x = pd.read_excel(io=xl_i, sheet_name=0, header=None)
data_j_x = pd.read_excel(io=xl_j, sheet_name=0, header=None)
data_k_x = pd.read_excel(io=xl_k, sheet_name=0, header=None)
data_l_x = pd.read_excel(io=xl_l, sheet_name=0, header=None)
data_m_x = pd.read_excel(io=xl_m, sheet_name=0, header=None)
data_n_x = pd.read_excel(io=xl_n, sheet_name=0, header=None)
data_o_x = pd.read_excel(io=xl_o, sheet_name=0, header=None)
data_p_x = pd.read_excel(io=xl_p, sheet_name=0, header=None)
data_q_x = pd.read_excel(io=xl_q, sheet_name=0, header=None)
data_r_x = pd.read_excel(io=xl_r, sheet_name=0, header=None)
data_s_x = pd.read_excel(io=xl_s, sheet_name=0, header=None)
data_t_x = pd.read_excel(io=xl_t, sheet_name=0, header=None)
data_u_x = pd.read_excel(io=xl_u, sheet_name=0, header=None)
data_v_x = pd.read_excel(io=xl_v, sheet_name=0, header=None)
data_w_x = pd.read_excel(io=xl_w, sheet_name=0, header=None)
data_x_x = pd.read_excel(io=xl_x, sheet_name=0, header=None)
data_y_x = pd.read_excel(io=xl_y, sheet_name=0, header=None)
data_z_x = pd.read_excel(io=xl_z, sheet_name=0, header=None)

data_a_y = pd.read_excel(io=xl_a, sheet_name=1, header=None)
data_b_y = pd.read_excel(io=xl_b, sheet_name=1, header=None)
data_c_y = pd.read_excel(io=xl_c, sheet_name=1, header=None)
data_d_y = pd.read_excel(io=xl_d, sheet_name=1, header=None)
data_e_y = pd.read_excel(io=xl_e, sheet_name=1, header=None)
data_f_y = pd.read_excel(io=xl_f, sheet_name=1, header=None)
data_g_y = pd.read_excel(io=xl_g, sheet_name=1, header=None)
data_h_y = pd.read_excel(io=xl_h, sheet_name=1, header=None)
data_i_y = pd.read_excel(io=xl_i, sheet_name=1, header=None)
data_j_y = pd.read_excel(io=xl_j, sheet_name=1, header=None)
data_k_y = pd.read_excel(io=xl_k, sheet_name=1, header=None)
data_l_y = pd.read_excel(io=xl_l, sheet_name=1, header=None)
data_m_y = pd.read_excel(io=xl_m, sheet_name=1, header=None)
data_n_y = pd.read_excel(io=xl_n, sheet_name=1, header=None)
data_o_y = pd.read_excel(io=xl_o, sheet_name=1, header=None)
data_p_y = pd.read_excel(io=xl_p, sheet_name=1, header=None)
data_q_y = pd.read_excel(io=xl_q, sheet_name=1, header=None)
data_r_y = pd.read_excel(io=xl_r, sheet_name=1, header=None)
data_s_y = pd.read_excel(io=xl_s, sheet_name=1, header=None)
data_t_y = pd.read_excel(io=xl_t, sheet_name=1, header=None)
data_u_y = pd.read_excel(io=xl_u, sheet_name=1, header=None)
data_v_y = pd.read_excel(io=xl_v, sheet_name=1, header=None)
data_w_y = pd.read_excel(io=xl_w, sheet_name=1, header=None)
data_x_y = pd.read_excel(io=xl_x, sheet_name=1, header=None)
data_y_y = pd.read_excel(io=xl_y, sheet_name=1, header=None)
data_z_y = pd.read_excel(io=xl_z, sheet_name=1, header=None)

data_a_z = pd.read_excel(io=xl_a, sheet_name=2, header=None)
data_b_z = pd.read_excel(io=xl_b, sheet_name=2, header=None)
data_c_z = pd.read_excel(io=xl_c, sheet_name=2, header=None)
data_d_z = pd.read_excel(io=xl_d, sheet_name=2, header=None)
data_e_z = pd.read_excel(io=xl_e, sheet_name=2, header=None)
data_f_z = pd.read_excel(io=xl_f, sheet_name=2, header=None)
data_g_z = pd.read_excel(io=xl_g, sheet_name=2, header=None)
data_h_z = pd.read_excel(io=xl_h, sheet_name=2, header=None)
data_i_z = pd.read_excel(io=xl_i, sheet_name=2, header=None)
data_j_z = pd.read_excel(io=xl_j, sheet_name=2, header=None)
data_k_z = pd.read_excel(io=xl_k, sheet_name=2, header=None)
data_l_z = pd.read_excel(io=xl_l, sheet_name=2, header=None)
data_m_z = pd.read_excel(io=xl_m, sheet_name=2, header=None)
data_n_z = pd.read_excel(io=xl_n, sheet_name=2, header=None)
data_o_z = pd.read_excel(io=xl_o, sheet_name=2, header=None)
data_p_z = pd.read_excel(io=xl_p, sheet_name=2, header=None)
data_q_z = pd.read_excel(io=xl_q, sheet_name=2, header=None)
data_r_z = pd.read_excel(io=xl_r, sheet_name=2, header=None)
data_s_z = pd.read_excel(io=xl_s, sheet_name=2, header=None)
data_t_z = pd.read_excel(io=xl_t, sheet_name=2, header=None)
data_u_z = pd.read_excel(io=xl_u, sheet_name=2, header=None)
data_v_z = pd.read_excel(io=xl_v, sheet_name=2, header=None)
data_w_z = pd.read_excel(io=xl_w, sheet_name=2, header=None)
data_x_z = pd.read_excel(io=xl_x, sheet_name=2, header=None)
data_y_z = pd.read_excel(io=xl_y, sheet_name=2, header=None)
data_z_z = pd.read_excel(io=xl_z, sheet_name=2, header=None)

a_X = get_value(data_a_x)
b_X = get_value(data_b_x)
c_X = get_value(data_c_x)
d_X = get_value(data_d_x)
e_X = get_value(data_e_x)
f_X = get_value(data_f_x)
g_X = get_value(data_g_x)
h_X = get_value(data_h_x)
i_X = get_value(data_i_x)
j_X = get_value(data_j_x)
k_X = get_value(data_k_x)
l_X = get_value(data_l_x)
m_X = get_value(data_m_x)
n_X = get_value(data_n_x)
o_X = get_value(data_o_x)
p_X = get_value(data_p_x)
q_X = get_value(data_q_x)
r_X = get_value(data_r_x)
s_X = get_value(data_s_x)
t_X = get_value(data_t_x)
u_X = get_value(data_u_x)
v_X = get_value(data_v_x)
w_X = get_value(data_w_x)
x_X = get_value(data_x_x)
y_X = get_value(data_y_x)
z_X = get_value(data_z_x)

a_Y = get_value(data_a_y)
b_Y = get_value(data_b_y)
c_Y = get_value(data_c_y)
d_Y = get_value(data_d_y)
e_Y = get_value(data_e_y)
f_Y = get_value(data_f_y)
g_Y = get_value(data_g_y)
h_Y = get_value(data_h_y)
i_Y = get_value(data_i_y)
j_Y = get_value(data_j_y)
k_Y = get_value(data_k_y)
l_Y = get_value(data_l_y)
m_Y = get_value(data_m_y)
n_Y = get_value(data_n_y)
o_Y = get_value(data_o_y)
p_Y = get_value(data_p_y)
q_Y = get_value(data_q_y)
r_Y = get_value(data_r_y)
s_Y = get_value(data_s_y)
t_Y = get_value(data_t_y)
u_Y = get_value(data_u_y)
v_Y = get_value(data_v_y)
w_Y = get_value(data_w_y)
x_Y = get_value(data_x_y)
y_Y = get_value(data_y_y)
z_Y = get_value(data_z_y)

a_Z = get_value(data_a_z)
b_Z = get_value(data_b_z)
c_Z = get_value(data_c_z)
d_Z = get_value(data_d_z)
e_Z = get_value(data_e_z)
f_Z = get_value(data_f_z)
g_Z = get_value(data_g_z)
h_Z = get_value(data_h_z)
i_Z = get_value(data_i_z)
j_Z = get_value(data_j_z)
k_Z = get_value(data_k_z)
l_Z = get_value(data_l_z)
m_Z = get_value(data_m_z)
n_Z = get_value(data_n_z)
o_Z = get_value(data_o_z)
p_Z = get_value(data_p_z)
q_Z = get_value(data_q_z)
r_Z = get_value(data_r_z)
s_Z = get_value(data_s_z)
t_Z = get_value(data_t_z)
u_Z = get_value(data_u_z)
v_Z = get_value(data_v_z)
w_Z = get_value(data_w_z)
x_Z = get_value(data_x_z)
y_Z = get_value(data_y_z)
z_Z = get_value(data_z_z)



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

make("train_a.xls", "a", a_X, a_Y, a_Z)
make("train_b.xls", "b", b_X, b_Y, b_Z)
make("train_c.xls", "c", c_X, c_Y, c_Z)
make("train_d.xls", "d", d_X, d_Y, d_Z)
make("train_e.xls", "e", e_X, e_Y, e_Z)
make("train_f.xls", "f", f_X, f_Y, f_Z)
make("train_g.xls", "g", g_X, g_Y, g_Z)
make("train_h.xls", "h", h_X, h_Y, h_Z)
make("train_i.xls", "i", i_X, i_Y, i_Z)
make("train_j.xls", "j", j_X, j_Y, j_Z)
make("train_k.xls", "k", k_X, k_Y, k_Z)
make("train_l.xls", "l", l_X, l_Y, l_Z)
make("train_m.xls", "m", m_X, m_Y, m_Z)
make("train_n.xls", "n", n_X, n_Y, n_Z)
make("train_o.xls", "o", o_X, o_Y, o_Z)
make("train_p.xls", "p", p_X, p_Y, p_Z)
make("train_q.xls", "q", q_X, q_Y, q_Z)
make("train_r.xls", "r", r_X, r_Y, r_Z)
make("train_s.xls", "s", s_X, s_Y, s_Z)
make("train_t.xls", "t", t_X, t_Y, t_Z)
make("train_u.xls", "u", u_X, u_Y, u_Z)
make("train_v.xls", "v", v_X, v_Y, v_Z)
make("train_w.xls", "w", w_X, w_Y, w_Z)
make("train_x.xls", "x", x_X, x_Y, x_Z)
make("train_y.xls", "y", y_X, y_Y, y_Z)
make("train_z.xls", "z", z_X, z_Y, z_Z)








