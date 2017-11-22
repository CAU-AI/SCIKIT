from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

xl = pd.ExcelFile("Music_style_train.xlsx")

data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)
data = np.array(data_excel.values)
answer = np.array(answer_excel.values).flatten().transpose()


train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=0.3)

# Standardzation befoare training

scaler = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0).fit(train_data.astype(float))
train_data = scaler.transform(train_data.astype(float))
test_data = scaler.transform(test_data.astype(float))

accuracy_list=[]
Max_accuracy = 0
for i in range(100):
    gnb = GaussianNB()
    train_model = gnb.fit(train_data, train_answer)
    test_pred = train_model.predict(test_data)

    correct_count = (test_pred == test_answer).sum()
    accuracy = correct_count / len(test_answer)
    accuracy_list.append(accuracy)
    if(Max_accuracy < accuracy):
        Max_accuracy = accuracy
    print("NB = " + str(accuracy))



print("Max = " + str(Max_accuracy))



