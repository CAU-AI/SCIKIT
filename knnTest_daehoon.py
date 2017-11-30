from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

xl = pd.ExcelFile("Breastcancer_train.xlsx")
#xl = pd.ExcelFile("Music_style_train.xlsx")
#xl = pd.ExcelFile("3D_handwriting_train.xlsx")
#xl = pd.ExcelFile("zoo.xlsx")
#xl = pd.ExcelFile("Carnicom.xlsx")

data_excel = pd.read_excel(io=xl, sheetname=0, header=None)
answer_excel = pd.read_excel(io=xl, sheetname=1, header=None)
data = np.array(data_excel.values)
answer = np.array(answer_excel.values).flatten().transpose()



def do_test():
    train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=0.2)
    #Standardzation before training (20~22)
    scaler = preprocessing.StandardScaler().fit(train_data.astype(float))  #astype(float) = 요소들 float형태로 바꾸기
    train_data = scaler.transform(train_data.astype(float))
    test_data = scaler.transform(test_data.astype(float))

    nbrs = KNeighborsClassifier(n_neighbors=5)
    train_model = nbrs.fit(train_data, train_answer)
    test_pred = train_model.predict(test_data)



    #Standardization
    scaler = preprocessing.StandardScaler().fit(train_data.astype(float))
    train_data_proc = scaler.transform(train_data.astype(float))
    test_data_proc = scaler.transform(test_data.astype(float))

    #PCA
    feature_size = 3;      #차원수
    pca = LDA(n_components=feature_size)
    # pca.fit(train_data)

    train_data_proc = pca.fit_transform(train_data, train_answer)
    test_data_proc = pca.fit_transform(test_data, test_answer)

    nbrs = KNeighborsClassifier(n_neighbors=5)
    train_model = nbrs.fit(train_data_proc, train_answer)
    test_pred_proc = train_model.predict(test_data_proc)


    #PCC
    import numpy as np
    feature_size = 9;
    corr_array = []
    for i in range(0, train_data.shape[1]):
        corr_array.append(np.corrcoef(train_data[:, i], train_answer)[0, 1])
    corr_array = np.square(corr_array)
    pcc_feature_idx = np.flip(np.argsort(corr_array), 0)
    train_data_proc = train_data[:, pcc_feature_idx[0:feature_size]]
    test_data_proc = test_data[:,pcc_feature_idx[0:feature_size]]

    #Comp Classifier

    nbrs = KNeighborsClassifier(n_neighbors=5)
    train_model = nbrs.fit(train_data_proc, train_answer)
    test_pred = train_model.predict(test_data_proc)

    correct_count = (test_pred == test_answer).sum()
    accuracy = correct_count / len(test_answer)
    print("Accuracy proc= " + str(accuracy))
    return accuracy


def do_loop(count):
    sum = 0
    for i in range(0, count):
        sum += do_test()
    print("Avg : " +str(sum / count))


do_loop(300)

