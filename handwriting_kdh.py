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

