import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from preprocess import data, dataset, X, y, X_name
pd.options.display.max_columns = None

import pickle
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.utils import resample
from scipy.stats import entropy

from classifiers import clf

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_train_name, X_test_name, y_train_name, y_test_name = train_test_split(X_name, y, test_size=0.1, shuffle=False)

def train(cl):
    cl.fit(X_train, y_train)
    X_test_scaled = sc.transform(X_test)
    y_pred = cl.predict(X_test_scaled)
    for i,key in enumerate(X_test_name['HomeTeam']):
        if y_pred[i] == 1:
            # print("WON" + key)
            d[key] += 1
        elif y_pred[i] == -1:
            # print("LOST" + key)
            d[key] += -1

    # print("Accuracy test and train: "+str(cl))
    # acc = (accuracy_score(y_test, y_pred))

    # return acc

lr = clf[0]
# mlp = clf[4]

d = {}
for i in X_test_name['HomeTeam']:
    d[i] = 0

train(lr)

print(dict(sorted(d.items(), key=lambda item: item[1])))