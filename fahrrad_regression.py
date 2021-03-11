# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 16:20:32 2021

@author: katha
"""
# To Do
# interaktionseffekte fehlen noch bei der Parameterinterpretation
# Sample Size muss reduziert werden
# Qualität des Modells muss noch anhand des test sets geprüft werden


# Info zu den Daten
# https://www.opengov-muenchen.de/dataset/raddauerzaehlstellen-muenchen/resource/211e882d-fadd-468a-bf8a-0014ae65a393?view_id=11a47d6c-0bc1-4bfa-93ea-126089b59c3d

#--------------------------------------------------------------------
# change directory
#--------------------------------------------------------------------
import os
#os.chdir(r"C:\Users\katha\OneDrive\Dokumente\GitHub\Fahrrad_Muenchen_2020")
os.chdir(r"C:\Users\Maryna\Documents\GitHub\Fahrrad_Muenchen_2020")

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
df = pd.read_excel(io=r'.\01_Tageweise-Auswertung_StandDez2020.xlsx', sheet_name="Original")
# take a look at the dataset
df = df[df['zaehlstelle']!='Kreuther']
df = df.reset_index(drop = True)
df['C(zaehlstelle)'] = df['zaehlstelle']

df['cbewoelkung'] = df['bewoelkung'] 
df.loc[df['bewoelkung'] <= 20, 'cbewoelkung'] = '0wenig_bewoelkung' 
df.loc[(df['bewoelkung'] > 20) & (df['bewoelkung'] <= 90), 'cbewoelkung'] = '1mittel_bewoelkung' 
df.loc[df['bewoelkung'] > 90, 'cbewoelkung'] = '2viel_bewoelkung' 

df['cniederschlag'] = df['niederschlag'] 
df.loc[df['niederschlag'] <= 0, 'cniederschlag'] = '0kein_niederschlag' 
df.loc[(df['niederschlag'] > 0) & (df['niederschlag'] <= 2.5), 'cniederschlag'] = '1wenig_niederschlag' 
df.loc[(df['niederschlag'] > 2.5) & (df['niederschlag'] <= 10), 'cniederschlag'] = '2mittel_niederschlag' 
df.loc[df['niederschlag'] > 10, 'cniederschlag'] = '3viel_niederschlag' 

df['csonnenstunden'] = df['sonnenstunden'] 
df.loc[df['sonnenstunden'] <= 2.5, 'csonnenstunden'] = '0wenig_sonnenstunden' 
df.loc[(df['sonnenstunden'] > 2.5) & (df['sonnenstunden'] <= 5), 'csonnenstunden'] = '1mittel_sonnenstunden' 
df.loc[df['sonnenstunden'] > 5, 'csonnenstunden'] = '2viel_sonnenstunden' 

df.head()
print(np.mean(df))
print(np.std(df))

#--------------------------------------------------------------------
# Optional: are our independent variables normally distributed?
#--------------------------------------------------------------------

sns.distplot(df.min_temp) # 
sns.distplot(df.sonnenstunden) # 
sns.distplot(df.niederschlag) # 
sns.distplot(df.bewoelkung) # 

# independent variables don't seem to be normally distributed. Standardization
# thus does not make sense.
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
# z-transform dependent variable
#zndf = scaler.fit_transform(ndf) 
#zdf_1 = pd.DataFrame(zndf, columns = ndf.columns)
#zdf_2 = df[['datum', 'C(zaehlstelle)', 'zaehlstelle','min_temp','max_temp','niederschlag','bewoelkung', 'sonnenstunden']]
#zdf = zdf_2.join(zdf_1)
#zdf.tail(10)
zdf = df
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
plt.scatter( zdf.niederschlag, zdf.gesamt3,  color='blue')
plt.xlabel("Niederschlag")
plt.ylabel("Fahrradaufkommen")
plt.show()

# Temperatur
plt.scatter( zdf.min_temp, zdf.gesamt3,  color='blue')
plt.xlabel("min_temp")
plt.ylabel("Fahrradaufkommen")
plt.show()

plt.scatter( zdf.max_temp, zdf.gesamt3,  color='blue')
plt.xlabel("max_temp")
plt.ylabel("Fahrradaufkommen")
plt.show()

# Sonnenstunden
plt.scatter( zdf.sonnenstunden, zdf.gesamt3,  color='blue')
plt.xlabel("sonnenstunden")
plt.ylabel("Fahrradaufkommen")
plt.show()


# Sonnenstunden
plt.scatter( zdf.bewoelkung, zdf.gesamt3,  color='blue')
plt.xlabel("bewölkung")
plt.ylabel("Fahrradaufkommen")
plt.show()


#--------------------------------------------------------------------
# split data
#--------------------------------------------------------------------

msk = np.random.rand(len(zdf)) < 0.2
train = zdf[msk]
test = zdf[~msk]

#--------------------------------------------------------------------
# Model selection: Multicollinearity
#--------------------------------------------------------------------
features = "+".join(['min_temp', 'max_temp', 'cniederschlag',    'cbewoelkung', 'csonnenstunden'])
y, X = dmatrices('gesamt3 ~ ' + features, train, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)

# throw out vif > 4
features = "+".join(['min_temp', 'cniederschlag',  'cbewoelkung', 'csonnenstunden'])
y, X = dmatrices('gesamt3 ~ ' + features, train, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
# we are thwoing out max temperature

features = "+".join(['min_temp', 'cniederschlag',  'cbewoelkung'])
y, X = dmatrices('gesamt3 ~ ' + features, train, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)

#--------------------------------------------------------------------
# Ignoring Zaehlstellen
#--------------------------------------------------------------------
# prepare model fitting
i = 0
iterations_log = ""
# 1 fit initial full model
rtrain = train[['gesamt3', 'min_temp', 'cniederschlag', 'cbewoelkung']] # get variables of interest
features = "*".join(rtrain.columns[1:])
y = 'gesamt3'
first_model = ols(y + '~' + features, data=rtrain).fit()  # fit complete model
# if we want to remove higher order interactions
names_0 = dict(first_model.params)
names = make_variable_list(list(names_0.keys()))
names = remove_higher_order_interactions(names) # get rid of 3was interactions
new_formula = make_formula(names)
best_model_so_far = ols(new_formula, data=rtrain).fit() 
#best_model_so_far.summary()

# 2 fit model with worst interaction removed
# select worst interaction effect
dict_pvalues = dict(best_model_so_far.pvalues[1:])
dict_pvalues_interactions = {i: dict_pvalues[i] for i in dict_pvalues.keys() if ':' in i}
maxPval = dict_pvalues_interactions[max(dict_pvalues_interactions)]
worst_param =  remove_brackets(max(dict_pvalues_interactions))
names.remove(worst_param)
iterations_log += "\n ####### \n" + str(i) +  "AIC: "+ str(best_model_so_far.aic) +"\nworst p: "+ str(maxPval) + "\nworst B: " + str(worst_param)
print("Character Variables (Dropped):" + str(worst_param))

    
# 3 fit next best model: worst parameter removed
i += 1
new_formula = make_formula(names)
next_best_model = ols(new_formula, data=rtrain).fit() # fit model without worst parameter
# compare models: is it better without worst parameter?
while next_best_model.bic < best_model_so_far.bic and len(dict_pvalues_interactions)> 0:
        iterations_log+= 'Dropped \n'
        best_model_so_far = next_best_model
        # remove worst interaction for next model
        dict_pvalues = dict(best_model_so_far.pvalues[1:])
        dict_pvalues_interactions = {i: dict_pvalues[i] for i in dict_pvalues.keys() if ':' in i}
        try: 
            maxPval = dict_pvalues_interactions[max(dict_pvalues_interactions)]
            worst_param =  remove_brackets(max(dict_pvalues_interactions))
            iterations_log += "\n ####### \n" + str(i) +  "AIC: "+ str(model.aic) +"\nworst p: "+ str(maxPval) + "\nworst B: " + str(worst_param)
            names.remove(worst_param)
            print("Character Variables (Dropped):" + str(worst_param))
            # 4 fit next model
            i += 1
            new_formula = make_formula(names)
            next_best_model = ols(new_formula, data=rtrain).fit() 
        except:
            print('No Interaction effects left')
final_model = best_model_so_far
final_model.summary()
print(iterations_log)
# Bewölkung + Sonnenstunden + Niederschlag + Min-Temp 
# + min_temp:sonnenstunden
# + min_temp:bewoelkung
# + min_temp:niederschlag

std_gesamt3 = np.std(train.gesamt3)
parameters = dict(final_model.params)
pvals = dict(final_model.pvalues)

# Interpretation
# Min Temp Main effect
sns.scatterplot(data=train, x="min_temp", y="gesamt3")
delta_temp = 10
B_temp = final_model.params[list(parameters.keys()).index('min_temp')]* delta_temp #+ final_model.params[list(parameters.keys()).index('min_temp:sonnenstunden')]*sonnenstunden*delta_temp
D_temp = ( B_temp) **3
print("Pro Tag sind mit jeder " + str(delta_temp) + "°C Temperatursteigerung " + str(D_temp) + " Fahrräder zusätzlich auf den Straßen, wenn alle anderen Bedingungen gleich bleiben")
# Min Temp Interaction effect
#ax = plt.axes(projection='3d')
#ax.scatter3D(xs = list(train.min_temp), zs = list(train.gesamt3), ys = list(train.sonnenstunden));
#delta_temp = 5
#sonnenstunden = 8
#B_temp = final_model.params[list(parameters.keys()).index('min_temp')]* delta_temp #+ final_model.params[list(parameters.keys()).index('min_temp:sonnenstunden')]*sonnenstunden*delta_temp
#D_temp = ( B_temp*std_gesamt3) **3
#print("Pro Tag sind mit jeder " + str(delta_temp) + "°C Temperatursteigerung " + str(D_temp) + " Fahrräder zusätzlich auf den Straßen, wenn alle anderen Bedingungen gleich bleiben und bei " + str(sonnenstunden) + "Sonnenstunden")

# Niederschlag Main Effect
sns.scatterplot(data=train, x="cniederschlag", y="gesamt3")
delta_niederschlag = 10
B_niederschlag = final_model.params[list(parameters.keys()).index('cniederschlag')] * delta_niederschlag
D_niederschlag = ( B_niederschlag) **3
print("Pro Tag sind mit jedem " + str(delta_niederschlag) + "mm mehr Regen " + str(D_niederschlag) + " Fahrräder weniger auf den Straßen, wenn alle anderen Bedingungen gleich bleiben")

# # Niederschlag Interaction effect
# ax = plt.axes(projection='3d')
# ax.scatter3D(xs = list(train.niederschlag), zs = list(train.gesamt3), ys = list(train.min_temp));
# delta_niederschlag = 10
# B_niederschlag = final_model.params[list(parameters.keys()).index('niederschlag')] * delta_niederschlag
# D_niederschlag = ( B_niederschlag*std_gesamt3) **3
# print("Pro Tag sind mit jedem " + str(delta_niederschlag) + "mm mehr Regen " + str(D_niederschlag) + " Fahrräder weniger auf den Straßen, wenn alle anderen Bedingungen gleich bleiben")

# Bewoelkung
sns.scatterplot(data=train, x="cbewoelkung", y="gesamt3")
delta_bewoelkung =1
B_bewoelkung = final_model.params[list(parameters.keys()).index('cbewoelkung[T.1mittel]')]* delta_bewoelkung
D_bewoelkung = ( B_bewoelkung) **3
print("Pro Tag sind mit jedem " + str(delta_bewoelkung) + " % mehr Bewoelkung " + str(D_bewoelkung) + " Fahrräder weniger auf den Straßen, wenn alle anderen Bedingungen gleich bleiben")


B_bewoelkung = final_model.params[list(parameters.keys()).index('cbewoelkung[T.2viel]')]* delta_bewoelkung 
D_bewoelkung = ( B_bewoelkung) **3
print("Pro Tag sind mit jedem " + str(delta_bewoelkung) + " % mehr Bewoelkung " + str(D_bewoelkung) + " Fahrräder weniger auf den Straßen, wenn alle anderen Bedingungen gleich bleiben")

# Main Effects
#sns.scatterplot(data=train, x="sonnenstunden", y="gesamt3")
#B_sonnenstunden = final_model.params[list(parameters.keys()).index('sonnenstunden')]
#D_sonnenstunden = ( B_sonnenstunden*std_gesamt3) **3
#print("Pro Tag sind mit jeder zusätzlichen Sonnenstunde " + str(D_sonnenstunden) + " Fahrräder zusätzlich auf den Straßen, wenn alle anderen Bedingungen gleich bleiben")

# with interaction effects
#ax = plt.axes(projection='3d')
#ax.scatter3D(xs = list(train.sonnenstunden), zs = list(train.gesamt3), ys = list(train.min_temp));
#B_sonnenstunden = final_model.params[list(parameters.keys()).index('sonnenstunden')] #+ final_model.params[list(parameters.keys()).index('min_temp:sonnenstunden')]*15
#D_sonnenstunden = ( B_sonnenstunden*std_gesamt3) **3
#print("Pro Tag sind mit jeder zusätzlichen Sonnenstunde " + str(D_sonnenstunden) + " Fahrräder zusätzlich auf den Straßen, wenn alle anderen Bedingungen gleich bleiben")





# #--------------------------------------------------------------------
# # Model Selection: backward selection
# #--------------------------------------------------------------------
# i = 0
# iterations_log = ""
# # 1 fit full model
# rtrain = train[['gesamt3', 'zaehlstelle', 'C(zaehlstelle)', 'min_temp', 'niederschlag', 'sonnenstunden']]
# features = "*".join(rtrain.columns[2:])
# y = 'gesamt3'
# best_model_so_far = ols(y + '~' + features, data=rtrain).fit() 
# maxPval = max(best_model_so_far.pvalues)
# iterations_log += "\n" + str(i) +  "AIC: "+ str(best_model_so_far.aic) +"\nworst p: "+ str(maxPval)


# # if we want to remove higher order interactions
# names_0 = dict(best_model_so_far.params)
# names_raw = list(names_0.keys())
# names = make_variable_list(names_raw)
# names = remove_higher_order_interactions(names)
# new_formula = make_formula(names)
# i += 1
# best_model_so_far = ols(new_formula, data=rtrain).fit() 
# maxPval = max(best_model_so_far.pvalues)
# iterations_log += "\n" + str(i) +  "AIC: "+ str(best_model_so_far.aic) +"\nworst p: "+ str(maxPval)


# # 2 remove parameters
# names_0 = dict(best_model_so_far.params)
# names_raw = list(names_0.keys())
# names = make_variable_list(names_raw)
# worst_param =  remove_brackets(names_raw[best_model_so_far.pvalues.argmax()])
# names.remove(worst_param)
# print("Character Variables (Dropped):" + str(worst_param))

    
# # 3 fit next best model and compare to full model
# i += 1
# new_formula = make_formula(names)
# model = ols(new_formula, data=rtrain).fit() 
# maxPval = max(model.pvalues)
# iterations_log += "\n" + str(i) +  "AIC: "+ str(model.aic) + "\n worst p: "+ str(maxPval)
# while model.bic < best_model_so_far.bic:
#         best_model_so_far = model
#         # remove parameter for next model
        
#         worst_param =  remove_brackets(names_raw[best_model_so_far.pvalues.argmax()])
#         names.remove(worst_param)
#         print("Character Variables (Dropped):" + str(worst_param))
#         # 4 fit next model
#         i += 1
#         new_formula = make_formula(names)
#         model = ols(new_formula, data=rtrain).fit() 
#         maxPval = max(model.pvalues)
#         iterations_log += "\n" + str(i) +  "AIC: "+ str(model.aic) + "\n worst p: "+ str(maxPval)

# final_model = best_model_so_far
# final_model.summary()
# print(iterations_log)
# #--------------------------------------------------------------------
# # interpreting the coefficients
# #--------------------------------------------------------------------
# # units of variables
# # min_temp & max_temp = °C
# # sonnenstunden = stunden
# # niederschlag: 
# # bewoelkung: %

# # Zaehlstelle
# std_gesamt3 = np.std(df.gesamt3)
# parameters = dict(final_model.params)
# pvals = dict(final_model.pvalues)
# # Main Effects
# ax = sns.boxplot(x="zaehlstelle", y="gesamt3", data=train)
# for z in set(zdf.zaehlstelle):
#     if z != 'Arnulf':
#         print("Pro Tag gibt es in " + z + ' ' + str(round(final_model.params[list(parameters.keys()).index('C(zaehlstelle)[T.' + z + ']')]*std_gesamt3 **3, 2)) + int(pvals.get('C(zaehlstelle)[T.' + z + ']')< 0.025) * "*" + " Fahrräder mehr als in Arnulf, wenn alle anderen Bedingungen gleich bleiben")

# # Sonnenstunden
# sns.scatterplot(data=train, x="sonnenstunden", y="gesamt3", hue="zaehlstelle")
# B_sonnenstunden = final_model.params[list(parameters.keys()).index('sonnenstunden')]
# D_sonnenstunden = ( B_sonnenstunden*std_gesamt3) **3
# print("Pro Tag sind mit jeder zusätzlichen Sonnenstunde " + str(D_sonnenstunden) + " Fahrräder zusätzlich auf den Straßen, wenn alle anderen Bedingungen gleich bleiben")

# # Temperatur
# sns.scatterplot(data=train, x="min_temp", y="gesamt3", hue="zaehlstelle")
# B_temp = final_model.params[list(parameters.keys()).index('min_temp')]
# D_temp = ( B_temp*std_gesamt3) **3
# print("Pro Tag sind mit jeder 1°C Temperatursteigerung " + str(D_temp) + " Fahrräder zusätzlich auf den Straßen, wenn alle anderen Bedingungen gleich bleiben")

# # Niederschlag
# sns.scatterplot(data=train, x="niederschlag", y="gesamt3", hue="zaehlstelle")
# B_niederschlag = final_model.params[list(parameters.keys()).index('niederschlag')]
# D_niederschlag = ( B_niederschlag*std_gesamt3) **3
# print("Pro Tag sind mit jedem 1mm mehr Regen " + str(D_niederschlag) + " Fahrräder weniger auf den Straßen, wenn alle anderen Bedingungen gleich bleiben")

# # Bewoelkung
# sns.scatterplot(data=train, x="bewoelkung", y="gesamt3", hue="zaehlstelle")
# B_bewoelkung = final_model.params[list(parameters.keys()).index('niederschlag')]
# D_bewoelkung = ( B_bewoelkung*std_gesamt3) **3
# print("Pro Tag sind mit jedem 1 % mehr Bewoelkung " + str(D_bewoelkung) + " Fahrräder weniger auf den Straßen, wenn alle anderen Bedingungen gleich bleiben")



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

