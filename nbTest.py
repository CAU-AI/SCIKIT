from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



f = open('result1.txt', 'w+t')
xl = pd.ExcelFile("Breastcancer_train.xlsx")
# xl = pd.ExcelFile("Music_style_train.xlsx")
demo_xl = pd.ExcelFile("demo_test.xlsx")

data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)
data = np.array(data_excel.values)
answer = np.array(answer_excel.values).flatten().transpose()

demo_excel = pd.read_excel(io=demo_xl, sheetname=0, header=None)
answer_demo_excel = pd.read_excel(io=demo_xl, sheetname=1, header=None)
demo_data = np.array(demo_excel.values)
#demo_answer = np.array(answer_demo_excel.values).flatten().transpose()

train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=0.2)
train_data = data
train_answer = answer

# Standardzation before training
scaler = preprocessing.Imputer(missing_values='NaN', strategy='median').fit(train_data.astype(float))
train_data = scaler.transform(train_data.astype(float))
test_data = scaler.transform(test_data.astype(float))
#demo_data = scaler.transform(demo_data.astype(float))

#gnb = GaussianNB()
#train_model = gnb.fit(train_data, train_answer)
#test_pred1 = train_model.predict(demo_data)

#correct_count = (test_pred1 == demo_answer).sum()
#accuracy1 = correct_count / len(demo_answer)
#print("Accuracy = " + str(accuracy1))

feature_size = 3
lda = LDA(n_components=feature_size)

train_data_proc = lda.fit(train_data, train_answer)
#test_data_proc = lda.fit_transform(test_data, test_answer)
#demo_data_proc = lda.fit_transform(demo_data, demo_answer)

#print("train_data_proc : ")
#print( np.array(train_data_proc).shape)

#gnb = GaussianNB()
#train_model = gnb.fit(train_data_proc, train_answer)
test_pred2 = train_data_proc.predict(demo_data)

#correct_count = (test_pred2 == demo_answer).sum()
#accuracy2 = correct_count / len(demo_answer)
#print("Accuracy proc= " + str(accuracy2))

#if accuracy1 > accuracy2:
#    demo_pred = test_pred1
#else:
demo_pred = test_pred2

for i in demo_pred:
    f.write(str(i) + '\n')
print("finish")