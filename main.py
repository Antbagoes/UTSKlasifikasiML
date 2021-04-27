import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import  DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import KFold

#-----------------------------------------------------------------------------------------------------------------------

#function for kfold Xvalidation
def kfoldScore(fold, model, x, Y):
    kf = KFold(n_splits=fold, random_state=None, shuffle=False)  # make kfold
    precisionValue = np.average(cross_val_score(model, x, Y, scoring='accuracy', cv=kf, n_jobs=-1))
    return precisionValue

#function for evaluate using manual split
def splitScore(yTest, yPred):
    return metrics.accuracy_score(yTest, yPred)

#-----------------------------------------------------------------------------------------------------------------------


#load data
col_names = ['wifi1', 'wifi2', 'wifi3', 'wifi4', 'wifi5', 'wifi6', 'wifi7', 'room']
dataset = pd.read_excel('dataset/dataset.xls', index_col=None, header=None, names=col_names)

print("Struktur Dataset : ",dataset.shape)    #check data structure
print("Isi dataset : \n",dataset.head())   #print data to check
print("\n----------------------------------------------------------------------\n")


#-----------------------------------------------------------------------------------------------------------------------

#preparing data
X = dataset.drop('room', axis=1)    #x = feature
y = dataset['room']                 #y = label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) # split 70:30, train:test


#-----------------------------------------------------------------------------------------------------------------------


#Training and Making Predictions

#DT
DT_Model = DecisionTreeClassifier()
DT_Model.fit(X_train, y_train)
y_predDT = DT_Model.predict(X_test)

#SVM
SVM_Model = SVC(gamma='auto')
SVM_Model.fit(X_train, y_train)
y_predSVM = SVM_Model.predict(X_test)

#RF
RF_Model = RandomForestClassifier(n_estimators=40)
RF_Model.fit(X_train, y_train)
y_predRF = SVM_Model.predict(X_test)


#-----------------------------------------------------------------------------------------------------------------------

#evaluation
dtSplit = splitScore(y_test, y_predDT)      # Score DT Using manual split
svmSplit = splitScore(y_test, y_predSVM)    # Score SVM Using manual split
rfSplit = splitScore(y_test, y_predDT)      # Score RF Using manual split

numFold = 5
dtKFold5 = kfoldScore(numFold, DT_Model, X, y)      # Score DT Using 5 kfold
svmKFold5 = kfoldScore(numFold, SVM_Model, X, y)    # Score DT Using 5 kfold
rfKFold5 = kfoldScore(numFold, RF_Model, X, y)      # Score DT Using 5 kfold

numFold = 10
dtKFold10 = kfoldScore(numFold, DT_Model, X, y)      # Score DT Using 10 kfold
svmKFold10 = kfoldScore(numFold, SVM_Model, X, y)    # Score DT Using 10 kfold
rfKFold10 = kfoldScore(numFold, RF_Model, X, y)      # Score DT Using 10 kfold

accuracyScore = {'Algorithm': ['dtSplit','svmSplit','rfSplit','Audi A4','dtKFold5','svmKFold5','rfKFold5','dtKFold10','svmKFold10'],
        'Score': [dtSplit,svmSplit,rfSplit,dtKFold5,svmKFold5,rfKFold5,dtKFold10,svmKFold10,rfKFold10]
        }

accuracyTable = pd.DataFrame(accuracyScore, columns = ['Algorithm', 'Score'])
accuracyTable['Rank'] = accuracyTable['Score'].rank(ascending=False)
print("Hasil Perbandingan Algoritma : \n",accuracyTable)


