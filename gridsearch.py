from preprocess import data, dataset, X, y
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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

from classifiers import clf

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

params = {'C':[1,10],'gamma':[1,0.1], 'kernel':['linear','rbf']}
svc = SVC()
clf = RandomizedSearchCV(svc , params , cv=3 , verbose=4 , n_jobs=6)
clf.fit(X_train , y_train)

print(clf.best_params_ , clf.best_score_)

params = {'min_samples_leaf':[2,4,6,8,10,12] , 'min_samples_split': [2,4,6,8,10], 'max_features': ['auto', 'sqrt', 'log2']}
rfc = RandomForestClassifier()
clf = GridSearchCV(rfc , params , cv=3 , verbose=4 , n_jobs=6)
clf.fit(X_train , y_train)

print(clf.best_params_ , clf.best_score_)

params = {
    'hidden_layer_sizes': [(256,),(50,50)],
    'activation': ['logistic','tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05 , 0.01 , 0.001],
    'learning_rate': ['constant','adaptive'],
}
rfc = MLPClassifier(max_iter=1000)
clf = RandomizedSearchCV(rfc , params , cv=3 , verbose=3 , n_jobs=6)
clf.fit(X_train_scaled , y_train)

print(clf.best_params_ , clf.best_score_)

params = {'base_estimator': [LR,GNB,DT,RF,MLP,SVM]}
abc = AdaBoostClassifier()
clf = RandomizedSearchCV(abc , params , cv=3 , verbose=3 , n_jobs=6)
clf.fit(X_train_scaled , y_train)

print(clf.best_params_ , clf.best_score_)

params = {'min_samples_split':[2,4,6,8,10] , 'min_samples_leaf': [1,2,3,4,5,6] , "max_features":['auto','sqrt','log2'] , 'max_depth':[2,3,4,5]}
abc = GradientBoostingClassifier()
clf = RandomizedSearchCV(abc , params , cv=3 , verbose=3 , n_jobs=6)
clf.fit(X_train_scaled , y_train)

print(clf.best_params_ , clf.best_score_)