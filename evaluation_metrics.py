import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from preprocess import data, dataset, X, y
pd.options.display.max_columns = None

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


def generate(cl):
    cl.fit(X_train, y_train)
    X_test_scaled = sc.transform(X_test)
    y_pred = cl.predict(X_test_scaled)
    print("Accuracy test and train: "+str(cl))
    acc = (accuracy_score(y_test, y_pred))

    print("Classification Report")
    print(classification_report(y_test,y_pred))

    print("Confusion Matrix")
    confusion = confusion_matrix(y_test,y_pred)
    print(confusion)
    sns.heatmap(confusion, square=True,annot=True,fmt ='d',cbar=False)
    plt.xlabel('Actual Result')
    plt.ylabel('Predicted Result')
    plt.show()

for i in clf[5:]:
    generate(i)
    
