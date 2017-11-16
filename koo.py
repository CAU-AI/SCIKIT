from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.decomposition import PCA
import math

#xl = pd.ExcelFile("Breastcancer_train.xlsx")
#xl = pd.ExcelFile("Music_style_train.xlsx")
#xl = pd.ExcelFile("3D_handwriting_train.xlsx")
#xl = pd.ExcelFile("zoo.xlsx")
xl = pd.ExcelFile("Music_style_train.xlsx")

data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)
data = np.array(data_excel.values)
answer = np.array(answer_excel.values).flatten().transpose()


train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=0.2)

'''
#Standardzation before training
scaler = preprocessing.Imputer(missing_values='NaN',strategy='median').fit(train_data.astype(float))
train_data = scaler.transform(train_data.astype(float))
test_data = scaler.transform(test_data.astype(float))
'''



count = 0
dis = [[0.0] * len(train_data) for x in range(len(test_data))]

for i in range(0,len(test_data)):						# row
	for z in range(0,len(train_data)):
		for j in range(0,len(test_data[i])):				# column
			if math.isnan(test_data[i][j]): 
				count += 1
			else:
				if math.isnan(train_data[z][j]): 
					count += 1
				else:
					dis[i][z] += abs(test_data[i][j] - train_data[z][j])
		count = 0
		print(dis[i][z])
			
		#	count+=1
		#	print(count)
min = 0
for i in range(0,len(test_data)):
	for z in range(0,len(train_data) - 1):
		print(dis[i][z])


'''
nbrs = KNeighborsClassifier(n_neighbors=4)
train_model = nbrs.fit(train_data, train_answer)
test_pred = train_model.predict(test_data)
correct_count = (test_pred == test_answer).sum()
accuracy = correct_count / len(test_answer)
print("Accuracy = " + str(accuracy))
'''


