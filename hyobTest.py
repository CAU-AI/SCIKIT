from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


xl = pd.ExcelFile("3D_handwriting_train.xlsx")

data_x = pd.read_excel(io=xl, sheet_name=0, header=None)
data_y = pd.read_excel(io=xl, sheet_name=1, header=None)
data_z = pd.read_excel(io=xl, sheet_name=2, header=None)
data_a = pd.read_excel(io=xl, sheet_name=3, header=None)

dataX = np.array(data_x.values)
dataY = np.array(data_y.values)
dataZ = np.array(data_z.values)
answer = np.array(data_a.values).flatten().transpose()

train_data, test_data, train_answer, test_answer = train_test_split(dataX, answer, test_size=0.2)

''''''''
lr = linear_model.LinearRegression()

y = dataX

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, boston.data, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


#Standardzation before training
'''
scaler = preprocessing.StandardScaler().fit(train_data.astype(float))
train_data = scaler.transform(train_data.astype(float))
test_data = scaler.transform(test_data.astype(float))
'''

'''
nbrs = KNeighborsClassifier(n_neighbors=5)
train_model = nbrs.fit(train_data, train_answer)
test_pred = train_model.predict(test_data)


correct_count = (test_pred == test_answer).sum()
accuracy = correct_count / len(test_answer)
print("Accuracy = " + str(accuracy))
'''

#Standardization
'''
scaler = preprocessing.StandardScaler().fit(train_data.astype(float))
train_data_proc = scaler.transform(train_data.astype(float))
test_data_proc = scaler.transform(test_data.astype(float))
'''


'''
#PCA

feature_size = 20;
pca = PCA(n_components=feature_size)
pca.fit(train_data)

train_data_proc = pca.transform(train_data)
test_data_proc = pca.transform(test_data)
'''



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




'''
#Comp Classifier

nbrs = KNeighborsClassifier(n_neighbors=5)
train_model = nbrs.fit(train_data_proc, train_answer)
test_pred = train_model.predict(test_data_proc)

correct_count = (test_pred == test_answer).sum()
accuracy = correct_count / len(test_answer)
print("Accuracy proc= " + str(accuracy))
'''

