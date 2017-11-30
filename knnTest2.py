from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.decomposition import PCA



print("start excel")
#xl = pd.ExcelFile("zoo.xlsx")
#xl = pd.ExcelFile("3D_handwriting_train.xlsx")
#xl = pd.ExcelFile("cardiotocogram_train.xlsx")
#xl = pd.ExcelFile("Music_style_train.xlsx")
xl = pd.ExcelFile("Carnicom.xlsx")

print("exit excel")

data_excel = pd.read_excel(io=xl, sheet_name=0, header=None)
answer_excel = pd.read_excel(io=xl, sheet_name=1, header=None)
data = np.array(data_excel.values)
answer = np.array(answer_excel.values).flatten().transpose()

train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=0.2)
print("exit split")
nbrs = KNeighborsClassifier(n_neighbors=5)
print(train_data)
train_model = nbrs.fit(train_data, train_answer)
print("exit fit")
test_pred = train_model.predict(test_data)
print("exit predict")
correct_count = (test_pred == test_answer).sum()
accuracy = correct_count / len(test_answer)
print("Accuracy = " + str(accuracy))

#pca

feature_size = 20
pca = PCA(n_components=feature_size)
pca.fit(train_data)

train_data_proc = pca.transform(train_data)
test_data_proc = pca.transform(test_data)