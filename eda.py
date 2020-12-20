import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from preprocess import data, dataset, X, y
pd.options.display.max_columns = None

print(data.shape)
print(dataset.shape)

print(dataset.head())
print(dataset.tail())

print(dataset.describe())

sns.pairplot(dataset[['HS','AS','HST','AST','HomeAttack','AwayAttack','HomeAdv','HTForm','ATForm','Result']])
plt.show()

f, ax = plt.subplots(figsize=(25,25))
corr = dataset.corr()
sns.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool),cmap = sns.diverging_palette(220,10,as_cmap=True),square = True,ax=ax,annot=True)
plt.show()

sns.countplot(x='FTHG', data=dataset)
plt.show()

sns.countplot(x='FTAG', data=dataset)
plt.show()

sns.countplot(x='Result', data=dataset)
plt.show()

plt.scatter(dataset['HS'],dataset['HST'])
plt.xlabel("Shots")
plt.ylabel("Shots on Target")
plt.show()

matches = np.arange(len(dataset))
plt.plot(matches,dataset['HS'])
plt.plot(matches,dataset['AS'])
plt.xlabel("Matches")
plt.ylabel("Shots")
plt.show()

plt.plot(matches,dataset['HF'])
plt.plot(matches,dataset['AF'])
plt.xlabel("Matches")
plt.ylabel("Fouls")
plt.show()