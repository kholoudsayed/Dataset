from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.tree import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.decomposition import PCA



def SVM_Classifier(X_Train,Y_Train,X_Test,Y_Test):
    print("----SVC Model----")
    svc = svm.SVC(kernel='linear', C=1).fit(X_Train, Y_Train)
    lin_svc = svm.LinearSVC(C=1).fit(X_Train, Y_Train)
    rbf_svc = svm.SVC(kernel='rbf', C=5).fit(X_Train, Y_Train)
    poly_svc = svm.SVC(kernel='poly', degree=2, C=5).fit(X_Train, Y_Train)
    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
        predictions = clf.predict(X_Test)
        accuracy = np.mean(predictions == Y_Test)
        print(accuracy)



def Decisiontree_Classifier(X_Train,Y_Train,X_Test,Y_Test):
    print("----Decision Tree Model----")
    clf = tree.DecisionTreeClassifier(max_depth=100)
    clf.fit(X_Train, Y_Train)
    y_prediction = clf.predict(X_Test)
    accuracy = np.mean(y_prediction == Y_Test) * 100
    print("The achieved accuracy using Decision Tree is " + str(accuracy))



def AdaBoost_Classifier (X_Train , Y_Train,X_Test,Y_Test):
    print("----AdaBoost Classifier Model----")

    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=300),
                             n_estimators=100,learning_rate=1,algorithm="SAMME.R")
    scaler = StandardScaler()
    scaler.fit(X_Train)

    X_Train = scaler.transform(X_Train)
    X_Test = scaler.transform(X_Test)

    bdt.fit(X_Train, Y_Train)
    y_prediction = bdt.predict(X_Test)
    accuracy = np.mean(y_prediction == Y_Test) * 100
    print("The Achieved Accuracy using Adaboost is " + str(accuracy))







def KNeighbors_Classifier(N_K ,X_Train , Y_Train,X_Test,Y_Test):
    print("----KNN Classifier----")
    knn = KNeighborsClassifier(n_neighbors=N_K)
    knn.fit(X_Train, Y_Train)
    y_pred = knn.predict(X_Test)
    print("The Achieved Accuracy using K-Nearest Neighbor Classifier :", metrics.accuracy_score(Y_Test, y_pred)*100)
    
    

