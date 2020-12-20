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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.utils import resample
from scipy.stats import entropy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

LR = LogisticRegression()
GNB = GaussianNB()
DT = DecisionTreeClassifier(criterion='entropy', 
                            max_leaf_nodes=40, 
                            min_samples_split=2, 
                            min_samples_leaf=2)
RF = RandomForestClassifier(criterion="entropy", 
                            min_samples_leaf=2)
MLP = MLPClassifier(alpha=0.04,
                    hidden_layer_sizes=(256,), 
                    activation='logistic', 
                    solver='sgd', 
                    learning_rate='adaptive',
                    max_iter=1000)
SVM = SVC(kernel='rbf', C=50, gamma=0.005)
ADA = AdaBoostClassifier(base_estimator=LR)
GB = GradientBoostingClassifier(max_features='sqrt',
                                min_samples_leaf=4,
                                min_samples_split=6)

clf = [LR, GNB, DT, RF, MLP, SVM, ADA, GB]

