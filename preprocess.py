import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from streak import seasons

dataset = pd.concat(seasons)

def outcome(row):
    '''Converts results (H,A or D) into numeric values'''
    if(row.FTR == 'H'):
        return 1
    elif(row.FTR == 'A'):
        return -1
    else:
        return 0

dataset["Result"] = dataset.apply(lambda match: outcome(match),axis=1)
data = dataset
dataset = dataset.dropna()

X = dataset[['HS','AS','HST','AST','HC','AC','IWH','IWD','IWA','WHH','WHD','WHA']]
y = dataset['Result']

X_name= dataset[['HomeTeam','AwayTeam', 'HS','AS','HST','AST','HC','AC','IWH','IWD','IWA','WHH','WHD','WHA']]
