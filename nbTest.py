from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

xl = pd.ExcelFile("Breastcancer_train.xlsx")
# xl = pd.ExcelFile("Music_style_train.xlsx")
demo_xl = pd.ExcelFile("demo_test.xlsx")

f = open('result.txt', 'w+t')

data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)
data = np.array(data_excel.values)
answer = np.array(answer_excel.values).flatten().transpose()

demo_excel = pd.read_excel(io=demo_xl, sheetname=0, header=None)
answer_demo_excel = pd.read_excel(io=demo_xl, sheetname=1, header=None)

train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=0.2)

# Standardzation before training
scaler = preprocessing.Imputer(missing_values='NaN', strategy='median').fit(train_data.astype(float))
train_data = scaler.transform(train_data.astype(float))
test_data = scaler.transform(test_data.astype(float))

gnb = GaussianNB()
train_model = gnb.fit(train_data, train_answer)
test_pred1 = train_model.predict(test_data)

correct_count = (test_pred1 == test_answer).sum()
accuracy1 = correct_count / len(test_answer)
print("Accuracy = " + str(accuracy1))

feature_size = 5
pca = LDA(n_components=feature_size)

train_data_proc = pca.fit_transform(train_data, train_answer)
test_data_proc = pca.fit_transform(test_data, test_answer)

gnb = GaussianNB()
train_model = gnb.fit(train_data_proc, train_answer)
test_pred2 = train_model.predict(test_data_proc)

correct_count = (test_pred2 == test_answer).sum()
accuracy2 = correct_count / len(test_answer)
print("Accuracy proc= " + str(accuracy2))

if accuracy1 > accuracy2:
    demo_pred = test_pred1
else:
    demo_pred = test_pred2

for i in demo_pred:
    f.write(str(demo_pred[i]) + '\n')
