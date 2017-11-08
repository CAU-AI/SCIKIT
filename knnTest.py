from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.decomposition import PCA

#xl = pd.ExcelFile("zoo.xlsx")
#xl = pd.ExcelFile("3D_handwriting_train.xlsx")
#xl = pd.ExcelFile("cardiotocogram_train.xlsx")
#xl = pd.ExcelFile("Music_style_train.xlsx")
xl = pd.ExcelFile("Carnicom.xlsx")

data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)
data = np.array(data_excel.values)
answer = np.array(answer_excel.values).flatten().transpose()

train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=0.2)

#Standardzation before training
'''
scaler = preprocessing.StandardScaler().fit(train_data.astype(float))
train_data = scaler.transform(train_data.astype(float))
test_data = scaler.transform(test_data.astype(float))
'''

nbrs = KNeighborsClassifier(n_neighbors=5)
train_model = nbrs.fit(train_data, train_answer)
test_pred = train_model.predict(test_data)


correct_count = (test_pred == test_answer).sum()
accuracy = correct_count / len(test_answer)
print("Accuracy = " + str(accuracy))

#Standardization
'''
scaler = preprocessing.StandardScaler().fit(train_data.astype(float))
train_data_proc = scaler.transform(train_data.astype(float))
test_data_proc = scaler.transform(test_data.astype(float))
'''

#PCA

feature_size = 20;
pca = PCA(n_components=feature_size)
pca.fit(train_data)

train_data_proc = pca.transform(train_data)
test_data_proc = pca.transform(test_data)

#PCC
'''
import numpy as np
feature_size = 500;
corr_array = []
for i in range(0, train_data.shape[1]):
    corr_array.append(np.corrcoef(train_data[:, i], train_answer)[0, 1])
corr_array = np.square(corr_array)
pcc_feature_idx = np.flip(np.argsort(corr_array), 0)
train_data_proc = train_data[:, pcc_feature_idx[0:feature_size]]
test_data_proc = test_data[:,pcc_feature_idx[0:feature_size]]
'''

#Comp Classifier

nbrs = KNeighborsClassifier(n_neighbors=5)
train_model = nbrs.fit(train_data_proc, train_answer)
test_pred = train_model.predict(test_data_proc)

correct_count = (test_pred == test_answer).sum()
accuracy = correct_count / len(test_answer)
print("Accuracy proc= " + str(accuracy))


