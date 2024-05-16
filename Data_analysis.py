# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 07:39:09 2024

@author: shubh
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as pgo
import tensorflow as tf 

###########################       Step 1       ################################
########################### Importing Data Set ################################

# Importing the dataset 
dataset = pd.read_csv('tips.csv')
#x = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, -2].values

# Printing the data set
print(dataset.head())

# Printing all the column names 
column_headers = list(dataset.columns.values)
print("The Column Header :", column_headers)

###########################       Step 2       ################################
###########################  Cleaning Data Set ################################


# Checking if data set has null values 
print(dataset.isnull().sum())

# Collecting all the null values present
all_rows = dataset[dataset.isnull().any(axis=1)] 

#deleting duplicate values
# Duplicate values 
dupilcate = dataset[dataset.duplicated()]

#deleting duplicate values
dataset.drop_duplicates(inplace=True)


# printing type of days, time
print(dataset.groupby('day').size())
print(dataset.groupby('time').size())


###############################################################################

# Effect of days on the tips 

# figure = pe.scatter(data_frame = dataset, x="total_bill",
#                     y="tip", size="size", color= "day", trendline="ols")
# figure.show()
fig_tip = px.scatter(dataset, x="total_bill", y="tip", 
                 size='size', color= "day", trendline="ols")
fig_tip.show()
fig_tip.write_image("Tip_vs_day.png")

fig = px.pie(dataset, values="total_bill", names="day")
fig.show()
fig.write_image("Tip_vs_day_Pi.png")




fig_tip = px.scatter(dataset, x="total_bill", y="tip", 
                 size='size', color= "sex", trendline="ols")
fig_tip.show()
fig_tip.write_image("Tip_vs_sex_scatter.png")

fig = px.pie(dataset, values="total_bill", names="sex")
fig.show()
fig.write_image("Tip_vs_sex_Pi.png")



fig_tip = px.scatter(dataset, x="total_bill", y="tip", 
                 size='size', color= "day", trendline="ols")
fig_tip.show()
fig_tip.write_image("Tip_vs_day_scatter.png")

fig = px.pie(dataset, values="total_bill", names="day")
fig.show()
fig.write_image("Tip_vs_day_Pi.png")


fig_tip = px.scatter(dataset, x="total_bill", y="tip", 
                 size='size', color= "time", trendline="ols")
fig_tip.show()
fig_tip.write_image("Tip_vs_time_scatter.png")

fig = px.pie(dataset, values="total_bill", names="time")
fig.show()
fig.write_image("Tip_vs_time_Pi.png")

fig_tip = px.scatter(dataset, x="total_bill", y="tip", 
                 size='size', color= "size", trendline="ols")
fig_tip.show()
fig_tip.write_image("Tip_vs_size_scatter.png")

fig = px.pie(dataset, values="total_bill", names="size")
fig.show()
fig.write_image("Tip_vs_size_Pi.png")


fig_tip = px.scatter(dataset, x="total_bill", y="tip", 
                 size='size', color= "smoker", trendline="ols")
fig_tip.show()
fig_tip.write_image("Tip_vs_smoker_scatter.png")

fig = px.pie(dataset, values="total_bill", names="smoker")
fig.show()
fig.write_image("Tip_vs_smoker_Pi.png")




fig = px.pie(dataset, values="total_bill", names="day")
fig.show()
fig.write_image("fig1.png")



###########################       Step 3       ################################
###########################  Feature Selction  ################################

# Feature selection 
x = np.array(dataset[['total_bill','sex','smoker','day','time','size']])
y = np.array(dataset['tip'])


# from sklearn.feature_selection import mutual_info_classif
# importance = mutual_info_classif(x,y)
# feat_importance = pd.Series(importance, dataset.columns[1: len(dataset.columns)-1])
# km = feat_importance.plot(kind='barh', color='teal',title="Information Gain")
# km.figure.savefig('Infomation_gain.png')

## as the number of featurres are less so we dont need feature selection

###########################       Step 4            ###########################
###########################  Categorical variables  ###########################


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x  = np.array(ct.fit_transform(x))

# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [7])], remainder='passthrough')
# x  = np.array(ct.fit_transform(x))




# labelencoder is for yes and no values 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,5] = le.fit_transform(x[:,5])
x[:,6] = le.fit_transform(x[:,6])
x[:,7] = le.fit_transform(x[:,7])





###########################       Step 5            ###########################
###########################    Data splitting       ###########################

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
###########################       Step 6            ###########################
###########################  Model Implimentation   ###########################



# Rrandom Forest
from sklearn.ensemble import RandomForestRegressor
regressorrdf = RandomForestRegressor(n_estimators=50,max_depth=3, random_state=0)
regressorrdf.fit(x_train,y_train)
y_pred_randtree = regressorrdf.predict(x_test)



############# evaluation of model with R2 model##############
from sklearn.metrics import r2_score

print('R2 score by Randomforest:',r2_score(y_test,y_pred_randtree))
#print('R2 score by ann:',r2_score(y_test,y_pred))

# K- validation
from sklearn.model_selection import cross_val_score
accuracy_k = cross_val_score(estimator= regressorrdf,X = x_train, y = y_train, cv = 10, n_jobs = -1) # use n_job =-1 to use cpus
print(' Average acuracy by K fold cross validation: ', accuracy_k.mean())
print('Std acuracy by K fold cross validation:',accuracy_k.std())
