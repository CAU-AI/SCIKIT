from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import PCA as pca

#test_size 별 accuracy 측정

xl = pd.ExcelFile("Breastcancer_train.xlsx")
data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)
data = np.array(data_excel.values)
answer = np.array(answer_excel.values).flatten().transpose()

range_list = np.arange(0.1, 1, 0.1)
max_accuracy = 0
max_test_size = 0

print("PCA nb")
for i in range_list:
    train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=i)
    gnb = GaussianNB()
    train_model = gnb.fit(train_data, train_answer)
    test_pred = train_model.predict(test_data)
    correct_count = (test_pred == test_answer).sum()
    accuracy = correct_count / len(test_answer)
    if max_accuracy < accuracy:
        max_accuracy = accuracy
        max_test_size = i
    print("(test_size : " + str(i) + ") data= " + str(correct_count) + " test_data = " + str(len(test_data))+ "\t\tAccuracy = " + str(accuracy))


print("\nPCA nb")
max_pca_accuracy = 0
max_pca_test_size = 0

data_PCA = np.array(pca.new_data1)
data_PCA = np.c_[data_PCA,pca.new_data2]

for i in range_list:
    train_data, test_data, train_answer, test_answer = train_test_split(data_PCA, answer, test_size=i)
    gnb = GaussianNB()
    train_model = gnb.fit(train_data, train_answer)
    test_pred = train_model.predict(test_data)
    correct_count = (test_pred == test_answer).sum()
    accuracy = correct_count / len(test_answer)
    if max_pca_accuracy < accuracy:
        max_pca_accuracy = accuracy
        max_pca_test_size = i
    print("(test_size : " + str(i) + ")  data = " + str(correct_count) + " test_data = " + str(
        len(test_data)) + "\t\tAccuracy = " + str(accuracy))

print("\nPCA : Max_accuracy's test_size = " + str(max_test_size) + "   MAX_accuracy = " + str(max_accuracy))
print("PCA : Max_accuracy's test_size = " + str(max_pca_test_size) + "   Max_accuracy = " + str(max_pca_accuracy))

