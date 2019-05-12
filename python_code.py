# -*- coding: utf-8 -*-
"""
Created on Mon May  6 00:25:58 2019

@author: Mitsiou
"""

# LOAN CASE CLASSIFICATION

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn import preprocessing

#-----------------------------------------------------------------------------

# preprocessing - APPLIES TO ALL
df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv")
df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], inplace=True)
df.columns = df.columns.str.lower()
df["effective_date"] = pd.to_datetime(df.effective_date)
df["due_date"] = pd.to_datetime(df.due_date)
df["education"] = df.education.replace(["college", "Bechalor"], ["College", "Bachelor"])
df["dayofweek"] = df.effective_date.dt.dayofweek
df["weekend"] = df.dayofweek.apply(lambda d: 1 if d > 3 else 0)
df["gender"] = df.gender.replace(["male", "female"], [0, 1])

Feature = df[['principal','terms','age','gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)

X = Feature
y = df['loan_status'].values #array

X = preprocessing.StandardScaler().fit(X).transform(X) #convert to array

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


#-----------------------------------------------------------------------------

# kNN ---------------------
# import library
from sklearn.neighbors import KNeighborsClassifier

# accuracy evaluation
from sklearn import metrics

# calculate accuracy for different kNNs
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

# train model and predict
for n in range(1,Ks):
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)  
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

# plotting the performance
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3x std'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (k)')
plt.tight_layout()
plt.show()

print( "kNN: Best accuracy is", mean_acc.max(), "with k =", mean_acc.argmax()+1) 

#-----------------------------------------------------------------------------

# Decision Tree ---------------------
# import library
from sklearn.tree import DecisionTreeClassifier

# calculate accuracy for different depths
Depths = 10
mean_acc = np.zeros((Depths-1))
std_acc = np.zeros((Depths-1))

# train model and predict
for n in range(3,Depths):
    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = n)
    drugTree.fit(X_train,y_train)
    predTree = drugTree.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, predTree)  
    std_acc[n-1] = np.std(predTree==y_test)/np.sqrt(predTree.shape[0])

# plotting the performance
plt.plot(range(1,Depths),mean_acc,'g')
plt.fill_between(range(1,Depths),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3x std'))
plt.ylabel('Accuracy ')
plt.xlabel('Tree Depth')
plt.tight_layout()
plt.show()

print( "Decision Tree: Best accuracy is", mean_acc.max(), "with depth =", mean_acc.argmax()+1) 

#-----------------------------------------------------------------------------
# Support Vector Machine ---------------------
# import library
from sklearn import svm

# transform y_train, y_test to numeric values
y_train = np.where(y_train == "PAIDOFF", 1, 0) 
y_test = np.where(y_test == "PAIDOFF", 1, 0) 

# calculate accuracy for different kernel
Kernels = ["linear", "poly", "rbf", "sigmoid"]
mean_acc = np.zeros(len(Kernels))
std_acc = np.zeros(len(Kernels))

# train model and predict
for kernel in Kernels:
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train) 
    yhat = clf.predict(X_test)
    mean_acc[Kernels.index(kernel)] = metrics.accuracy_score(y_test, yhat)  
    std_acc[Kernels.index(kernel)] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

# plotting the performance
plt.plot(Kernels,mean_acc,'g')
plt.fill_between(Kernels,mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3x std'))
plt.ylabel('Accuracy ')
plt.xlabel('Kernel')
plt.tight_layout()
plt.show()

print( "SVM: Best accuracy is", mean_acc.max(), "for kernel", Kernels[mean_acc.argmax()]) 

#-----------------------------------------------------------------------------
# Logistic Regression ---------------------
# import library
from sklearn.linear_model import LogisticRegression

# calculate accuracy for different solver
Solvers = ['liblinear', 'newton-cg', 'sag', 'lbfgs']
mean_acc = np.zeros(len(Solvers))
std_acc = np.zeros(len(Solvers))

# train model and predict
for solver in Solvers:
    LR = LogisticRegression(C=0.01, solver=solver).fit(X_train,y_train)
    yhat = LR.predict(X_test)
    mean_acc[Solvers.index(solver)] = metrics.accuracy_score(y_test, yhat)  
    std_acc[Solvers.index(solver)] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

# plotting the performance
plt.plot(Solvers,mean_acc,'g')
plt.fill_between(Solvers,mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3x std'))
plt.ylabel('Accuracy ')
plt.xlabel('Solver')
plt.tight_layout()
plt.show()

print( "Logistic Regression: Best accuracy is", mean_acc.max(), "for solver", Solvers[mean_acc.argmax()]) 


#-----------------------------------------------------------------------------
# preprocessing of the REAL test set - APPLIES TO ALL
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

test_df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv")

test_df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], inplace=True)
test_df.columns = test_df.columns.str.lower()
test_df["effective_date"] = pd.to_datetime(test_df.effective_date)
test_df["due_date"] = pd.to_datetime(test_df.due_date)
test_df["education"] = test_df.education.replace(["college", "Bechalor"], ["College", "Bachelor"])
test_df["dayofweek"] = test_df.effective_date.dt.dayofweek
test_df["weekend"] = test_df.dayofweek.apply(lambda d: 1 if d > 3 else 0)
test_df["gender"] = test_df.gender.replace(["male", "female"], [0, 1])

Feature = test_df[['principal','terms','age','gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(test_df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)

X = Feature
X_test = preprocessing.StandardScaler().fit(X).transform(X)
y_test = test_df.loan_status.values

performance = {}

#-----------------------------------------------------------------------------

# kNN model evaluation
neigh = KNeighborsClassifier(n_neighbors = 7).fit(X_test, y_test)
yhat=neigh.predict(X_test)

performance["kNN"] = {"Jaccard": jaccard_similarity_score(y_test, yhat), 
                      "F1-score": f1_score(y_test, yhat, average="weighted"),
                      "LogLoss": None}

#-----------------------------------------------------------------------------

# Decision Tree model evaluation
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 6)
drugTree.fit(X_test, y_test)
predTree = drugTree.predict(X_test)

performance["Decision Tree"] = {"Jaccard": jaccard_similarity_score(y_test, predTree), 
                                "F1-score": f1_score(y_test, predTree, average="weighted"),
                                "LogLoss": None}

#-----------------------------------------------------------------------------

# SVM evaluation
y_test = np.where(y_test == "PAIDOFF", 1, 0) 

clf = svm.SVC(kernel="linear")
clf.fit(X_test, y_test) 
yhat = clf.predict(X_test)

performance["SVM"] = {"Jaccard": jaccard_similarity_score(y_test, yhat), 
                      "F1-score": f1_score(y_test, yhat, average="weighted"),
                      "LogLoss": None}

#-----------------------------------------------------------------------------

# Logistic Regression evaluation
from sklearn.metrics import log_loss

LR = LogisticRegression(C=0.01, solver='newton-cg').fit(X_test, y_test)
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

performance["Logistic Regression"] = {"Jaccard": jaccard_similarity_score(y_test, yhat), 
                                      "F1-score": f1_score(y_test, yhat, average="weighted"),
                                      "LogLoss": log_loss(y_test, yhat_prob)}
#-----------------------------------------------------------------------------
algorithmPerformance = pd.DataFrame(data=performance).T.round(3)
algorithmPerformance.index.name = "algorithm"
algorithmPerformance = algorithmPerformance.loc[:, ["Jaccard", "F1-score", "LogLoss"]]
print(algorithmPerformance)
#-----------------------------------------------------------------------------





















































































