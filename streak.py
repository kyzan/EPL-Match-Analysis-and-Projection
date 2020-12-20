import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

features = ['HomeTeam','AwayTeam','FTHG', 'FTAG', 'FTR','HS','AS','HST','AST','HC','AC','HF','AF','HY','AY','HR','AR','IWH','IWD','IWA','WHH','WHD','WHA']

s03 = pd.read_csv('EPL_DATA/2003 - 2003.csv')
s04 = pd.read_csv('EPL_DATA/2004 - 2004.csv')

seasons  = [0 for i in range(20)]
for i in range(20):
    if i>4 or i<3:    
        index = 'EPL_DATA/' + str(2000 + i) + '.csv'
        seasons[i] = pd.read_csv(index)[features]
seasons[3] = s03[features]
seasons[4] = s04[features]

for s in seasons:
    HomeAttack = [0 for i in range(len(s))]
    AwayAttack = HomeAttack
    s['HomeAttack'] = s['HS']
    s['AwayAttack'] = s['AS']
    s['HomeAdv'] = 0
    s['HTForm'] = 0
    s['ATForm'] = 0
    s['redhome'] = 0
    s['redaway'] = 0

#attacking strength
def AttackSt(season):
    dict={}
    HAS=[]
    AAS=[]
    for i, row in season.iterrows():
        if row['HomeTeam'] not in dict:
            dict[row['HomeTeam']] = 0
        if row['AwayTeam'] not in dict:
            dict[row['AwayTeam']] = 0
    for i, row in season.iterrows():
        HAS.append(dict[row['HomeTeam']])
        AAS.append(dict[row['AwayTeam']])
        dict[row['HomeTeam']] += row['HS']
        dict[row['AwayTeam']] += row['AS']
    season['HomeAttack'] = np.array(HAS)/max(HAS)
    season['AwayAttack'] = np.array(AAS)/max(AAS)
for s in seasons:
    AttackSt(s)
#homeadv
def HomeAdv(season):
    dict={}
    HAS=[]
    for i, row in season.iterrows():
        if row['HomeTeam'] not in dict:
            dict[row['HomeTeam']] = 0
    for i, row in season.iterrows():
        HAS.append(dict[row['HomeTeam']])
        dict[row['HomeTeam']] += 2.3 + ((row['FTHG']-row['FTAG']))
    season['HomeAdv'] = np.array(HAS)

for s in seasons:
    HomeAdv(s)

def redcards(season):
    dict={}
    HAS=[]
    AAS=[]
    for i, row in season.iterrows():
        if row['HomeTeam'] not in dict:
            dict[row['HomeTeam']] = 0
        if row['AwayTeam'] not in dict:
           dict[row['AwayTeam']] = 0
    for i, row in season.iterrows():
        HAS.append(dict[row['HomeTeam']])
        AAS.append(dict[row['AwayTeam']])
        x = dict[row['HomeTeam']]
        y = dict[row['AwayTeam']]
        if(row['HR'] > 0):
            x -= row['HR']
        else:
            x = 0
        if (row['AR'] > 0):
            y -= row['AR']
        else:
            y = 0
    season['redhome'] = np.array(HAS)
    season['redaway'] = np.array(AAS)

for s in seasons:
    redcards(s)

#form
def Form(season):
    dict={}
    HAS=[]
    AAS=[]
    for i, row in season.iterrows():
        if row['HomeTeam'] not in dict:
            dict[row['HomeTeam']] = 0
        if row['AwayTeam'] not in dict:
           dict[row['AwayTeam']] = 0
    for i, row in season.iterrows():
        HAS.append(dict[row['HomeTeam']])
        AAS.append(dict[row['AwayTeam']])
        x = dict[row['HomeTeam']]
        y = dict[row['AwayTeam']]
        if(row['FTR']=='H' and x<7):
            dict[row['HomeTeam']] += 0.8
        if (row['FTR']=='A' and x>-7):
            dict[row['HomeTeam']] -= 0.5
        if(row['FTR']=='A' and y<7):
            dict[row['AwayTeam']] += 0.8
        if (row['FTR']=='H' and y>-7):
            dict[row['AwayTeam']] -= 0.5
    season['HTForm'] = np.array(HAS)
    season['ATForm'] = np.array(AAS)

for s in seasons:
    Form(s)






