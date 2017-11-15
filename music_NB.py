from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import preprocessing


xl = pd.ExcelFile("Music_style_train.xlsx")
data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)
data = np.array(data_excel.values)
answer = np.array(answer_excel.values).flatten().transpose()

train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=0.2)


scaler = preprocessing.Imputer(missing_values='NaN').fit(data.astype(float))
train_data = scaler.transform(train_data.astype(float))
test_data = scaler.transform(test_data.astype(float))

gnb = GaussianNB()
train_model = gnb.fit(train_data, train_answer)
test_pred = train_model.predict(test_data)

correct_count = (test_pred == test_answer).sum()
accuracy = correct_count / len(test_answer)
print("Accuracy = " + str(accuracy))
