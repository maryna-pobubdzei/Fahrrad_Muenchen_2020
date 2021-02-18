# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 16:20:32 2021

@author: katha
"""

# regression von fahrradaufkommen

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from patsy import dmatrices
import seaborn as sns
from scipy import stats
import stepwise_selection as ss

# import data
df = pd.read_excel(io=r'C:\Users\katha\OneDrive\Dokumente\Projekte\Fahrrad\01_Tageweise-Auswertung_StandDez2020.xlsx', sheet_name="Original")
# take a look at the dataset
df.head()

#normalverteilt?
np.mean(df.gesamt)
np.std(df.gesamt)  
sns.distplot(df.gesamt) # clear right skww
sns.distplot(np.log(df.gesamt+1)) # log transform
sns.distplot((df.gesamt)**(1./2.)) # sqrt transform
sns.distplot((df.gesamt)**(1./3.)) # cube transform
# best transform is cube transform (based on visual inspection)
df['gesamt3'] = df.gesamt ** (1./3.)

# standardize variables
ndf = df[['gesamt', 'gesamt3','min_temp','max_temp','niederschlag','bewoelkung', 'sonnenstunden']]
scaler = StandardScaler()
# fit and transform the data
zndf = scaler.fit_transform(ndf) 

zdf_1 = pd.DataFrame(zndf, columns = ndf.columns)
zdf_2 = df[['datum', 'zaehlstelle']]
zdf = zdf_2.join(zdf_1)
zdf.tail(10)

sns.distplot(zdf.gesamt) # cube transform
sns.distplot(zdf.gesamt3) # cube transform
k2 , p = stats.normaltest(zdf.gesamt3) # überhaupt nicht normalverteilt :( 


# interactions with fahrradnutzung
plt.scatter( zdf.niederschlag,zdf.gesamt3,  color='blue')
plt.xlabel("Niederschlag")
plt.ylabel("Fahrradaufkommen")
plt.show()

plt.scatter( zdf.min_temp    ,zdf.gesamt3,  color='blue')
plt.xlabel("min_temp")
plt.ylabel("Fahrradaufkommen")
plt.show()

plt.scatter( zdf.max_temp, zdf.gesamt3,  color='blue')
plt.xlabel("max_temp")
plt.ylabel("Fahrradaufkommen")
plt.show()

plt.scatter( zdf.sonnenstunden    ,zdf.gesamt,  color='blue')
plt.xlabel("sonnenstunden")
plt.ylabel("Fahrradaufkommen")
plt.show()


# split data
msk = np.random.rand(len(zdf)) < 0.8
train = zdf[msk]
test = zdf[~msk]

# Model selection: choose among main effects
from statsmodels.formula.api import ols
# 1. Multicollinearity between main effects
fit = ols('gesamt3 ~ C(zaehlstelle)  + min_temp + max_temp + niederschlag + sonnenstunden', data=train).fit() 
fit.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor
features = "+".join(['min_temp', 'max_temp', 'niederschlag',    'bewoelkung', 'sonnenstunden'])
y, X = dmatrices('gesamt ~ ' + features, zdf, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)

# throw out vif > 4
features = "+".join(['min_temp', 'niederschlag',  'bewoelkung', 'sonnenstunden'])
y, X = dmatrices('gesamt ~ ' + features, zdf, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
# we are thwoing out max temperature

# 2. 
fit_full = ols('gesamt3 ~ C(zaehlstelle)  * min_temp  * niederschlag * sonnenstunden', data=train).fit() 
fit_full.summary()
fit_full.aic
 
fit_main = ols('gesamt3 ~ C(zaehlstelle)  + min_temp  + niederschlag + sonnenstunden', data=train).fit() 
fit_main.aic


final_vars, iterations_logs = ss.forwardSelection(test[['min_temp', 'max_temp', 'sonnenstunden', 'zaehlstelle']], test.gesamt3)

