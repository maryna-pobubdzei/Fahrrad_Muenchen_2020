# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 16:20:32 2021

@author: katha
"""

# Info zu den Daten
# https://www.opengov-muenchen.de/dataset/raddauerzaehlstellen-muenchen/resource/211e882d-fadd-468a-bf8a-0014ae65a393?view_id=11a47d6c-0bc1-4bfa-93ea-126089b59c3d

#--------------------------------------------------------------------
# change directory
#--------------------------------------------------------------------
import os
os.chdir(r"C:\Users\katha\OneDrive\Dokumente\GitHub\Fahrrad_Muenchen_2020")
######### change directory #############
#--------------------------------------------------------------------
# import modules
#--------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from patsy import dmatrices
import seaborn as sns
from scipy import stats
from additional_functions import *
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

#--------------------------------------------------------------------
# import data
#--------------------------------------------------------------------
df = pd.read_excel(io=r'C:\Users\katha\OneDrive\Dokumente\Projekte\Fahrrad\01_Tageweise-Auswertung_StandDez2020.xlsx', sheet_name="Original")
# take a look at the dataset
df = df[df['zaehlstelle']!='Kreuther']
df = df.reset_index(drop = True)
df['C(zaehlstelle)'] = df['zaehlstelle']
#df['log_niederschlag'] = np.log(df['niederschlag']+1)
df.head()
print(np.mean(df))
print(np.std(df))

#--------------------------------------------------------------------
# Optional: are our independent variables normally distributed?
#--------------------------------------------------------------------
sns.distplot(np.log(zdf.niederschlag + 1)) #

sns.distplot(df.min_temp) # 
sns.distplot(df.sonnenstunden) # 
sns.distplot(df.niederschlag) # 

#--------------------------------------------------------------------
# Requirement: is our dependent variable normally distributed?
#--------------------------------------------------------------------
print(np.mean(df.gesamt))
print(np.std(df.gesamt)) 
sns.distplot(df.gesamt) # clear right skww
sns.distplot(np.log(df.gesamt+1)) # log transform
sns.distplot((df.gesamt)**(1./2.)) # sqrt transform
sns.distplot((df.gesamt)**(1./3.)) # cube transform
# best transform is cube transform (based on visual inspection)
df['gesamt3'] = df.gesamt ** (1./3.)
ndf = df[['gesamt', 'gesamt3']]
scaler = StandardScaler()
# fit and transform the data
zndf = scaler.fit_transform(ndf) 
zdf_1 = pd.DataFrame(zndf, columns = ndf.columns)
zdf_2 = df[['datum', 'C(zaehlstelle)', 'zaehlstelle','min_temp','max_temp','niederschlag','bewoelkung', 'sonnenstunden']]
zdf = zdf_2.join(zdf_1)
zdf.tail(10)
sns.distplot(zdf.gesamt) # cube transform
sns.distplot(zdf.gesamt3) # cube transform
k2 , p = stats.normaltest(zdf.gesamt3) # überhaupt nicht normalverteilt :( 
print(p)
ax = sns.boxplot(x="zaehlstelle", y="gesamt3", data=zdf)
print(np.mean(zdf))
print(np.std(zdf))



#--------------------------------------------------------------------
# First visual inspection of association with Fahrradaufkommen
#--------------------------------------------------------------------
# Zählstelle
ax = sns.boxplot(x="zaehlstelle", y="gesamt3", data=zdf)


# Niederschlag
plt.scatter( (zdf.niederschlag),zdf.gesamt3,  color='blue')
plt.xlabel("Niederschlag")
plt.ylabel("Fahrradaufkommen")
plt.show()

# Temperatur
plt.scatter( zdf.min_temp    ,zdf.gesamt3,  color='blue')
plt.xlabel("min_temp")
plt.ylabel("Fahrradaufkommen")
plt.show()

plt.scatter( zdf.max_temp, zdf.gesamt3,  color='blue')
plt.xlabel("max_temp")
plt.ylabel("Fahrradaufkommen")
plt.show()

# Sonnenstunden
plt.scatter( zdf.sonnenstunden    ,zdf.gesamt3,  color='blue')
plt.xlabel("sonnenstunden")
plt.ylabel("Fahrradaufkommen")
plt.show()

#--------------------------------------------------------------------
# split data
#--------------------------------------------------------------------

msk = np.random.rand(len(zdf)) < 0.8
train = zdf[msk]
test = zdf[~msk]

#--------------------------------------------------------------------
# Model selection: Multicollinearity
#--------------------------------------------------------------------
fit = ols('gesamt3 ~ C(zaehlstelle)  + min_temp + max_temp + niederschlag + sonnenstunden', data=train).fit() 
fit.summary()
features = "+".join(['min_temp', 'max_temp', 'niederschlag',    'bewoelkung', 'sonnenstunden'])
y, X = dmatrices('gesamt3 ~ ' + features, train, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)

# throw out vif > 4
features = "+".join(['min_temp', 'niederschlag',  'bewoelkung', 'sonnenstunden'])
y, X = dmatrices('gesamt3 ~ ' + features, train, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
# we are thwoing out max temperature

#--------------------------------------------------------------------
# Model Selection: backward selection
#--------------------------------------------------------------------
i = 0
iterations_log = ""
# 1 fit full model
rtrain = train[['gesamt3', 'zaehlstelle', 'C(zaehlstelle)', 'min_temp', 'niederschlag', 'sonnenstunden']]
features = "*".join(rtrain.columns[2:])
y = 'gesamt3'
best_model_so_far = ols(y + '~' + features, data=rtrain).fit() 
maxPval = max(best_model_so_far.pvalues)
iterations_log += "\n" + str(i) +  "AIC: "+ str(best_model_so_far.aic) +"\nworst p: "+ str(maxPval)


# if we want to remove higher order interactions
names_0 = dict(best_model_so_far.params)
names_raw = list(names_0.keys())
names = make_variable_list(names_raw)
names = remove_higher_order_interactions(names)
new_formula = make_formula(names)
i += 1
best_model_so_far = ols(new_formula, data=rtrain).fit() 
maxPval = max(best_model_so_far.pvalues)
iterations_log += "\n" + str(i) +  "AIC: "+ str(best_model_so_far.aic) +"\nworst p: "+ str(maxPval)


# 2 remove parameters
names_0 = dict(best_model_so_far.params)
names_raw = list(names_0.keys())
names = make_variable_list(names_raw)
worst_param =  remove_brackets(names_raw[best_model_so_far.pvalues.argmax()])
names.remove(worst_param)
print("Character Variables (Dropped):" + str(worst_param))

    
# 3 fit next best model and compare to full model
i += 1
new_formula = make_formula(names)
model = ols(new_formula, data=rtrain).fit() 
maxPval = max(model.pvalues)
iterations_log += "\n" + str(i) +  "AIC: "+ str(model.aic) + "\n worst p: "+ str(maxPval)
while model.bic < best_model_so_far.bic:
        best_model_so_far = model
        # remove parameter for next model
        
        worst_param =  remove_brackets(names_raw[best_model_so_far.pvalues.argmax()])
        names.remove(worst_param)
        print("Character Variables (Dropped):" + str(worst_param))
        # 4 fit next model
        i += 1
        new_formula = make_formula(names)
        model = ols(new_formula, data=rtrain).fit() 
        maxPval = max(model.pvalues)
        iterations_log += "\n" + str(i) +  "AIC: "+ str(model.aic) + "\n worst p: "+ str(maxPval)

final_model = best_model_so_far
final_model.summary()
print(iterations_log)
#--------------------------------------------------------------------
# interpreting the coefficients
#--------------------------------------------------------------------
# units of variables
# min_temp & max_temp = °C
# sonnenstunden = stunden
# niederschlag: 
# bewoelkung: %


std_gesamt3 = np.std(df_rm.gesamt3)
parameters = dict(final_model.params)
pvals = dict(final_model.pvalues)
# Main Effects
ax = sns.boxplot(x="zaehlstelle", y="gesamt3", data=train)
for z in set(zdf.zaehlstelle):
    if z != 'Arnulf':
        print("Pro Tag gibt es in " + z + ' ' + str(round(final_model.params[list(parameters.keys()).index('C(zaehlstelle)[T.' + z + ']')]*std_gesamt3 **3, 2)) + int(pvals.get('C(zaehlstelle)[T.' + z + ']')< 0.025) * "*" + " Fahrräder mehr als in Arnulf, wenn alle anderen Bedingungen gleich bleiben")

sns.scatterplot(data=df_rm, x="sonnenstunden", y="gesamt", hue="zaehlstelle")
sns.scatterplot(data=train, x="sonnenstunden", y="gesamt3", hue="zaehlstelle")
B_sonnenstunden = (final_model.params[5]*np.std(df_rm.sonnenstunden)*std_gesamt3) **3
print("Pro Tag sind mit jeder zusätzlichen Sonnenstunde " + str(B_sonnenstunden) + " Fahrräder zusätzlich auf den Straßen, wenn alle anderen Bedingungen gleich bleiben")

#sns.scatterplot(data=df_rm, x="min_temp", y="gesamt", hue="zaehlstelle")
sns.scatterplot(data=train, x="min_temp", y="gesamt3", hue="zaehlstelle")
B_min_temp = (final_model.params[10]*np.std(df_rm.min_temp)*std_gesamt3) **3
print("Pro Tag sind mit jeder 1°C Temperatursteigerung " + str(B_min_temp) + " Fahrräder zusätzlich auf den Straßen, wenn alle anderen Bedingungen gleich bleiben")


#sns.scatterplot(data=df_rm, x="niederschlag", y="gesamt", hue="zaehlstelle")
sns.scatterplot(data=train, x="niederschlag", y="gesamt3", hue="zaehlstelle")
B_min_temp = (final_model.params[10]*np.std(df_rm.min_temp)*std_gesamt3) **3
print("Pro Tag sind mit jeder 1°C Temperatursteigerung " + str(B_min_temp) + " Fahrräder zusätzlich auf den Straßen, wenn alle anderen Bedingungen gleich bleiben")


sns.scatterplot(data=train, x="niederschlag", y="gesamt3", hue="zaehlstelle")
(np.exp(final_model.params[17]*std_gesamt3)-1) **3 # niederschlag



###############
# test data set 
###############

#from sklearn.metrics import r2_score

#test_x = np.asanyarray(test[['ENGINESIZE']])
#test_y = np.asanyarray(test[['CO2EMISSIONS']])
#score = final_model.predict(test)
#print("R2-score: %.2f" % r2_score(test[['gesamt3']] , score) )




    
#--------------------------------------------------------------------
# time course 
#--------------------------------------------------------------------
df[['datum', 'min_temp', 'max_temp']].set_index('datum').plot()
df[['datum', 'min_temp', 'sonnenstunden']].set_index('datum').plot()
df[['datum', 'min_temp', 'niederschlag']].set_index('datum').plot()
df[['datum', 'min_temp', 'bewoelkung']].set_index('datum').plot()

