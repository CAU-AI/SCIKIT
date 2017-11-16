from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.decomposition import PCA


xl = pd.ExcelFile("zoo.xlsx")
#xl = pd.ExcelFile("Carnicom.xlsx")

data_excel = pd.read_excel(io=xl, sheet_name=0, header=None)
answer_excel = pd.read_excel(io=xl, sheet_name=1, header=None)
data = np.array(data_excel.values)
answer = np.array(answer_excel.values).flatten().transpose()


train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=0.2)

gnb = GaussianNB()
train_model = gnb.fit(train_data, train_answer)
test_pred = train_model.predict(test_data)

correct_count = (test_pred == test_answer).sum()
accuracy = correct_count / len(test_answer)
print("Accuracy = " + str(accuracy))


print(train_data)
print(np.array(train_data).shape)
feature_size = 3;
pca = PCA(n_components=feature_size)
pca.fit(train_data)

train_data_proc = pca.transform(train_data)
test_data_proc = pca.transform(test_data)


print("train_data_proc : ")
print( np.array(train_data_proc).shape)

gnb = GaussianNB()
train_model = gnb.fit(train_data_proc, train_answer)
test_pred = train_model.predict(test_data_proc)

correct_count = (test_pred == test_answer).sum()
accuracy = correct_count / len(test_answer)
print("Accuracy proc= " + str(accuracy))
