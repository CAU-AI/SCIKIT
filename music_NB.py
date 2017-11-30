from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


xl = pd.ExcelFile("Music_style_train.xlsx")
data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)
data = np.array(data_excel.values)
answer = np.array(answer_excel.values).flatten().transpose()

predict_list=[]
def NB(feature):
    train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=0.2)

    scaler = preprocessing.Imputer(missing_values='NaN',strategy="mean").fit(data.astype(float))
    train_data = scaler.transform(train_data.astype(float))
    test_data = scaler.transform(test_data.astype(float))

    feature_size = feature  # 차원수
    pca = PCA(n_components=feature_size)
    pca.fit(train_data)

    train_data_proc = pca.transform(train_data)
    test_data_proc = pca.transform(test_data)

    gnb = GaussianNB()
    train_model = gnb.fit(train_data, train_answer)
    test_pred = train_model.predict(test_data)
    #print(test_pred)
    #print(test_answer)
    correct_count = (test_pred == test_answer).sum()
    accuracy = correct_count / len(test_answer)
    print("Accuracy = " + str(accuracy))
    predict_list.append(accuracy)
    return accuracy

range_num = 10
x=range(0,range_num )
accuracy_list=[]
#for i in range(range_num):
for j in range(0, 360, 1):
    accuracy_list.append( NB(j))
    #print("Max_Accuracy = (" + str(j) + "일때)" + str(max(predict_list)))


'''
plt.plot(x, predict_list, 'go')
plt.show()

'''
'''
plt.plot(range(0,360,20), accuracy_list, 'go')
plt.show()
'''