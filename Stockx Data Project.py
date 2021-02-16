#!/usr/bin/env python
# coding: utf-8

# # StockX Data Project: Attempting to Predict Sale Prices

# This project was inspired by my personal interest in the sneaker game, as well as the work conducted by Sameen Salam in his project last year at the Institute for Advanced Analytics. Using this inspiration, I hoped to find some significant results and practice my modeling abilities through this project.
# 
# The main sections of this notebook include exploring the data, engineering new features, different modeling approaches, and final conclusions/results.

# # 1. Exploratory Data Analysis

# We first need to pull in our stocks data set and import any libraries needed for the analysis. The data consists of a random sample of all Off-White and Yeezy 350 sales between the end of 2017 to beginning of 2019. You can view the full details about the 2019 StockX Data Challenge at the link here: https://stockx.com/news/the-2019-data-contest/

# In[63]:


# Importing the necessary libraries for the entire file
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import regressors
from regressors import stats as reg
import requests

# Direct to the folder with the files for importing
os.chdir("C:\\Users\\jorda\\OneDrive\\Documents\\IAA Files and Documents\\Side Project Files\\StockX")


# In[64]:


# Load in our data set
shoes = pd.read_csv("StockX-Data-Contest-2019-3.csv")


# In[65]:


# Take a glance at our data columns
shoes.head(n=10)


# Our data currently includes 8 columns:
#     - the date the shoe was ordered
#     - the brand of the shoe
#     - the actual name of the shoe
#     - the sale price aka the price the shoe resold for
#     - the retail price aka the first price for the shoe when it came out
#     - the date the shoe was released
#     - the size of the shoe
#     - the state that the buyer bought the shoe from

# In[66]:


# Cleaning the data to have our dollar values to be numeric
shoes[shoes.columns[3:5]] = shoes[shoes.columns[3:5]].replace('[\$,]', '', regex=True).astype(float)

# Transforming date variable to date time values
shoes['Order Date'] = pd.to_datetime(shoes['Order Date'])
shoes['Release Date'] = pd.to_datetime(shoes['Release Date'])


# In[67]:


# Check for missing values
shoes.isnull().sum()


# Luckily, we have no missing values in our data

# In[68]:


# Check the summary stats for each column
shoes.describe()


# We only have 3 numeric values that generate these summary stats, but it helps give us a better idea of the range of values these columns take on

# In[69]:


# Checking the types of our data
shoes.dtypes


# All of our data types look good now for what the column should represent

# In[70]:


# Plotting some distributions of our target variable
shoes['Sale Price'].hist(bins=20, grid=False, xlabelsize=12, ylabelsize=12, color='green')
plt.xlabel("Sale Price", fontsize=15)
plt.ylabel("Frequency",fontsize=15)
plt.title("Histogram of Our Target: Sale Price", fontsize=15)
plt.text(2000, 20000, r'$\mu=446, std=255$')


# Based on our histogram plot, we can see our target is right skewed, meaning we have a higher number of smaller sales prices and fewer number of larger sales prices. This will probably effect our assumption of normality when we conduct a linear regression down the line.

# In[71]:


# Plotting some distributions of shoe sizes
plt.hist(shoes['Shoe Size'], bins=20, color='grey')
plt.title('Histogram of Shoe Sizes', fontsize=15)
plt.xlabel('Shoe Size', fontsize=15)
plt.ylabel('Frequency', fontsize=15)


# From our histogram, we can see that in our sample of data people are buying sizes 9-12 the most often, which confirms what we typically see when new shoes come out

# In[72]:


# Plotting the change over time of sale prices
plt.plot(shoes['Order Date'].sub(shoes['Release Date'], axis=0)/np.timedelta64('1','D') , shoes['Sale Price'], color='green')
plt.title('Change in Sale Price Over Time', fontsize=15)
plt.xlabel('Days from Release Date', fontsize=15)
plt.ylabel('Sale Price', fontsize=15)


# Not the prettiest chart, but this allows us to see whether or not shoes are decreasing in sale price as the time from when they were released increases. We can see here that it does trend slightly up as you go up until about 2 years, and then it begins to decrease.

# ## 2. Feature Engineering

# In[73]:


# Feature engineering variables that I think may be predictive of sale price
shoes['Price Difference'] = shoes['Sale Price'] - shoes['Retail Price']

shoes['Price_Ratio'] = shoes['Sale Price'] / shoes['Retail Price']

shoes['Difference_in_Date'] = shoes['Order Date'].sub(shoes['Release Date'], axis=0)/np.timedelta64('1','D') 

shoes['Order_Day'] = shoes['Order Date'].dt.day_name()

shoes['Release_Day'] = shoes['Release Date'].dt.day_name()

shoes['Order Month'] = shoes['Order Date'].dt.month_name()

shoes['Release Month'] = shoes['Release Date'].dt.month_name()

shoes["Color_Black"] = shoes['Sneaker Name'].apply(lambda x : 1 if 'Black' in x.split("-") else 0)

shoes["Color_Blue"] = shoes['Sneaker Name'].apply(lambda x : 1 if 'Blue' in x.split("-") else 0)

shoes["Yeezy_Boost350"] = shoes['Sneaker Name'].apply(lambda x : 1 if 'Boost' in x.split("-") else 0)

shoes["Air_Max"] = shoes['Sneaker Name'].apply(lambda x : 1 if 'Max' in x.split("-") else 0)

shoes["Presto"] = shoes['Sneaker Name'].apply(lambda x : 1 if 'Presto' in x.split("-") else 0)

shoes["Zoom_Fly"] = shoes['Sneaker Name'].apply(lambda x : 1 if 'Zoom' in x.split("-") else 0)

shoes["Blazer"] = shoes['Sneaker Name'].apply(lambda x : 1 if 'Blazer' in x.split("-") else 0)

shoes["Air_Force"] = shoes['Sneaker Name'].apply(lambda x : 1 if 'Force' in x.split("-") else 0)

shoes["Jordan_1"] = shoes['Sneaker Name'].apply(lambda x : 1 if 'Jordan' in x.split("-") else 0)

shoes["Yeezy_V2"] = shoes['Sneaker Name'].apply(lambda x : 1 if 'V2' in x.split("-") else 0)


# There are many additional features that can be engineered, however, I opted for these to start based on prior knowledge

# In[74]:


# Double check that our new columns are the right data type
shoes.dtypes


# In[75]:


# Double check our summary stats look good
shoes.describe()


# In[76]:


# Double check we have no missing values
shoes.isnull().sum()


# In[77]:


# Exporting the updated stockx data to use in a Tableau dashboard (go check it out!)
shoes.to_csv("stockx_data_updated.csv", index=False)


# ## 3. Modeling

# For our modeling, we are going to start with a linear regression model since we are trying to predict the sale price of a shoe (a continuous target). We will do one model for yeezys and one model for off-whites since they may have different predictors that contribute to an increase or decrease in sale price. We will then move to a random forest regressor to see if we can more accurately predict sale prices using different models. We will use the MAPE or mean absolute precent error to compare the results of our models. We will use MAPE as it has good interpretability and is easier to understand in a business sense. We also know that are target is strictly positive (cannot have negative sale prices) so that makes MAPE a good metric candidate. 
# 
# For linear regression, we need to meet 4 assumptions that we will evaluate after running the model:
#     1. There exists a linear relationship between x and y
#     2. The residuals are independent
#     3. The residuals have constant variance
#     4. The residuals of the model are normaly distributed

# In[78]:


# Need to first split data into yeezy and off white data to build separate models for them
shoes_yeezy = shoes[shoes['Brand'] == ' Yeezy']
shoes_ow = shoes[shoes['Brand'] == 'Off-White']


# In[79]:


# Dropping columns that are captured by other variables for each of our data sets
shoes_yeezy = shoes_yeezy.drop(columns=['Shoe Size', 'Difference_in_Date', 'Release Date', 'Price_Ratio', 'Order Date', 'Price Difference', 'Order Month', 'Order_Day', 'Sneaker Name', 'Buyer Region', 'Brand', 'Air_Max', 'Presto', 'Zoom_Fly', 'Blazer', 'Air_Force', 'Jordan_1', 'Yeezy_Boost350'])
shoes_ow = shoes_ow.drop(columns=['Shoe Size', 'Difference_in_Date', 'Release Date', 'Price_Ratio', 'Order Date', 'Price Difference', 'Order Month', 'Order_Day', 'Sneaker Name', 'Buyer Region', 'Brand', 'Yeezy_Boost350', 'Yeezy_V2'])


# Because we would be predicting sale price BEFORE a shoe comes out, we need to eliminate a lot of our features since we would not know this information before resell. We also eliminate some variables based on potential multicollinearity issues.

# In[80]:


# Creating dummy variables for our categorical variables so the model can handle them
shoes_yeezy = pd.get_dummies(shoes_yeezy)
shoes_ow = pd.get_dummies(shoes_ow)


# In[81]:


# Create objects x and y to separate our predictors from the target for both data sets (predicting sale price)
# Yeezy variables
x_yeezy = shoes_yeezy.drop('Sale Price', axis=1)
y_yeezy = shoes_yeezy['Sale Price']

# Off-white variables
x_ow = shoes_ow.drop('Sale Price', axis=1)
y_ow = shoes_ow['Sale Price']

# Run train_test_split to create a train/test data set for the x and y variables for our yeezy and off-white data sets
# We opt for a 80/20 split because we have a decent amount of data but want to make sure we have enough to train on
x_train_yeezy, x_test_yeezy, y_train_yeezy, y_test_yeezy = train_test_split(x_yeezy, y_yeezy, test_size = 0.20, random_state=50)
x_train_ow, x_test_ow, y_train_ow, y_test_ow = train_test_split(x_ow, y_ow, test_size = 0.20, random_state=50)


# ### We can now start with running the Yeezy Linear Regression Model

# In[82]:


# Looking at our correlations between predictors and target for yeezy data to make sure there are no issues
shoes_yeezy.corr(method='pearson')


# In[83]:


# Create the linear model: lm1 for our yeezy data
lm1_yeezy = sm.OLS(y_train_yeezy, x_train_yeezy).fit()

# Inspect the results
print(lm1_yeezy.summary())


# Lets examine some of the results from this model:
# 
# <br> **R Squared/Adjusted R Squared:** the model explains 64% of the variability of the response data around its mean
# <br> **Durbin Watson:** A durbin watson value of 1.98 indicates we have no issue of autocorrelation in our model
# <br> **Kurtosis:** A value of 21.821 indicates we have very heavy "tails" in our distribution (to be expected)
# <br> **AIC:** 679400; The likelihood of the model to predict/estimate future values (used for model comparison)
# <br> **BIC:** 679400; Similar to our AIC value, used for model comparison
# <br> **P Values:** All of our p values for our variables seem significant except for the Release Day of Tuesday (we are using a 0.005 cutoff for our p values due to our sample size and wanting a fair amount of evidence)

# In[84]:


# Getting test predictions
test_preds_yeezy1 = lm1_yeezy.predict(x_test_yeezy)

# Calculating the residuals
residuals_yeezy1 = test_preds_yeezy1 - y_test_yeezy

# Making QQ-plot of residuals (to check again for normality)
sm.qqplot(residuals_yeezy1,line="s")


# In[85]:


#Calculate the MAPE
lm1_yeezy_mape = np.mean(100 * abs((residuals_yeezy1/y_test_yeezy)))
lm1_yeezy_mape


# ### Now we can repeat the same process for Off-White Linear Regression

# In[86]:


#Looking at the correlation values for each predictor relative to each other and the target
shoes_ow.corr(method='pearson')


# In[87]:


# Create the linear model: lm1 for our off-white data
lm1_ow = sm.OLS(y_train_ow, x_train_ow).fit()

# Inspect the results
print(lm1_ow.summary())


# Lets examine some of the results from this model:
# 
# <br> **R Squared/Adjusted R Squared:** the model explains 77.1% of the variability of the response data around its mean
# <br> **Durbin Watson:** A durbin watson value of 2.007 indicates we have no issue of autocorrelation in our model
# <br> **Kurtosis:** A value of 20.478 indicates we have very heavy "tails" in our distribution (to be expected)
# <br> **AIC:** 288800; The likelihood of the model to predict/estimate future values (used for model comparison)
# <br> **BIC:** 289000; Similar to our AIC value, used for model comparison
# <br> **P Values:** All of our p values for our variables seem significant except for the Release Month of August, July, and September (we are using a 0.005 cutoff for our p values due to our sample size and wanting a fair amount of evidence)

# In[88]:


# Getting test predictions
test_preds_ow1 = lm1_ow.predict(x_test_ow)

# Calculating residuals
residuals_ow1 = test_preds_ow1 - y_test_ow

# Making QQ-plot of residuals
sm.qqplot(residuals_ow1,line="s")


# In[89]:


# Calculate the MAPE
lm1_ow_mape = np.mean(100 * abs((residuals_ow1/y_test_ow)))
lm1_ow_mape


# ### Second Linear Regression Model for Yeezy using a Transformed Target

# In[90]:


# Multiple Linear Regression with sqrt transform (transform our target for ow and yeezy)
y_train_yeezy_sqrt = np.sqrt(y_train_yeezy)
y_train_ow_sqrt = np.sqrt(y_train_ow)
y_test_yeezy_sqrt = np.sqrt(y_test_yeezy)
y_test_ow_sqrt = np.sqrt(y_test_ow)


# In[91]:


# Create the linear model: lm2 for our yeezy data
lm2_yeezy = sm.OLS(y_train_yeezy_sqrt, x_train_yeezy).fit()

# Inspect the results
print(lm2_yeezy.summary())


# Lets examine some of the results from this model:
# 
# <br> **R Squared/Adjusted R Squared:** the model explains 63.1% of the variability of the response data around its mean (it has gone down from our first model)
# <br> **Durbin Watson:** A durbin watson value of 1.987 indicates we have no issue of autocorrelation in our model
# <br> **Kurtosis:** A value of 9.636 indicates we have heavy tails still, but it has gone down a lot
# <br> **AIC:** 244100; The likelihood of the model to predict/estimate future values (used for model comparison)
# <br> **BIC:** 244100; Similar to our AIC value, used for model comparison
# <br> **P Values:** All of our p values for our variables seem significant except for the Release Day of Tuesday still (we are using a 0.005 cutoff for our p values due to our sample size and wanting a fair amount of evidence)

# In[92]:


# Getting test predictions
test_preds_yeezy = lm2_yeezy.predict(x_test_yeezy)

# Calculating residuals
residuals_yeezy2 = np.exp(test_preds_yeezy) - y_test_yeezy_sqrt

# Making QQ-plot of the residuals
sm.qqplot(residuals_yeezy2,line="s")


# In[93]:


# Calculate the MAPE: On average, the square root transformed linear model is 188821312199.28% off in predicting our sale price (YIKES)
lm2_sqrty_mape = np.mean(100 * abs(residuals_yeezy2/y_test_yeezy_sqrt))
lm2_sqrty_mape


# ### Second Linear Regression Model for Off-White using a Transformed Target

# In[94]:


# Create the linear model: lm2 for our off-white data
lm2_ow = sm.OLS(y_train_ow_sqrt, x_train_ow).fit()

# Inspect the results
print(lm2_ow.summary())


# Lets examine some of the results from this model:
# 
# <br> **R Squared/Adjusted R Squared:** the model explains 79.7% of the variability of the response data around its mean (it has gone up from our first model)
# <br> **Durbin Watson:** A durbin watson value of 2.009 indicates we have no issue of autocorrelation in our model
# <br> **Kurtosis:** A value of 8.739 indicates we have heavy tails still, but it has gone down a lot
# <br> **AIC:** 105300; The likelihood of the model to predict/estimate future values (used for model comparison)
# <br> **BIC:** 105300; Similar to our AIC value, used for model comparison
# <br> **P Values:** All of our p values for our variables seem significant except for the Release Month of April, November, and October (we are using a 0.005 cutoff for our p values due to our sample size and wanting a fair amount of evidence)

# In[95]:


# Getting test predictions
test_preds_ow = lm2_ow.predict(x_test_ow)

# Calculating residuals
residuals_ow2 = np.exp(test_preds_ow) - y_test_ow_sqrt

# Making QQ-plot of the residuals
sm.qqplot(residuals_ow2,line="s")


# In[96]:


#Calculate the MAPE: On average, the square root transformed linear model is WAY off in predicting our sale price (Yikes again!)
lm2_sqrtow_mape = np.mean(100 * abs(residuals_ow2/y_test_ow_sqrt))
lm2_sqrtow_mape


# Based on the results above, we fail to meet the normality assumption even with transformations to the target. We can try using a different model that does not have those same strict assumptions that linear regression does

# ### Random Forest Model for Yeezy and Off-White 

# In[97]:


# Need to split our test into a validation as well to avoid overfitting issues
x_test_yeezy_rf, x_valid_yeezy_rf, y_test_yeezy_rf, y_valid_yeezy_rf = train_test_split(x_test_yeezy,y_test_yeezy,test_size = 0.5)
x_test_ow_rf, x_valid_ow_rf, y_test_ow_rf, y_valid_ow_rf = train_test_split(x_test_ow,y_test_ow,test_size = 0.5)


# We want to tune a few hyperparameters for our random forest to try and produce the most accurate results on our data. We will conduct separate tuning for yeezy and off-white data as the hyperparameters may vary.

# In[98]:


# Starting with tuning the number of estimators hyper parameter and basing it off of the best MAPE value (yeezy)
n_estimators = [5, 10, 50, 100, 200, 350, 500, 800, 1000]
mape_results = []

for estimator in n_estimators:
    rf = RandomForestRegressor(n_estimators=estimator, random_state=50)
    rf.fit(x_train_yeezy, y_train_yeezy)
    valid_pred = rf.predict(x_valid_yeezy_rf)
    valid_error = valid_pred - y_valid_yeezy_rf
    mape = np.mean(100 * abs((valid_error/valid_pred)))
    mape_results.append(mape)

print(mape_results)


# In[99]:


# Starting with tuning the number of estimators hyper parameter and basing it off of the best MAPE value (off white)
n_estimators = [5, 10, 50, 100, 200, 350, 500, 800, 1000]
mape_results = []

for estimator in n_estimators:
    rf = RandomForestRegressor(n_estimators=estimator, random_state=50)
    rf.fit(x_train_ow, y_train_ow)
    valid_pred = rf.predict(x_valid_ow_rf)
    valid_error = valid_pred - y_valid_ow_rf
    mape = np.mean(100 * abs((valid_error/valid_pred)))
    mape_results.append(mape)

print(mape_results)


# In[100]:


# Now we can tune the max depth hyper parameter and base it off best MAPE value (yeezy)
max_depth = [5, 10, 25, 40, 50, 75, 100]
mape_results = []

for depth in max_depth:
    rf = RandomForestRegressor(max_depth=depth, random_state=50)
    rf.fit(x_train_yeezy, y_train_yeezy)
    valid_pred = rf.predict(x_valid_yeezy_rf)
    valid_error = valid_pred - y_valid_yeezy_rf
    mape = np.mean(100 * abs((valid_error/valid_pred)))
    mape_results.append(mape)

print(mape_results)


# In[101]:


# Now we can tune the max depth hyper parameter and base it off best MAPE (off-white)
max_depth = [5, 10, 25, 40, 50, 75, 100]
mape_results = []

for depth in max_depth:
    rf = RandomForestRegressor(max_depth=depth, random_state=50)
    rf.fit(x_train_ow, y_train_ow)
    valid_pred = rf.predict(x_valid_ow_rf)
    valid_error = valid_pred - y_valid_ow_rf
    mape = np.mean(100 * abs((valid_error/valid_pred)))
    mape_results.append(mape)

print(mape_results)


# In[102]:


# Lastly, we can tune our max features hyper parameter to determine the best MAPE value (yeezy)
max_features = ['auto', 'sqrt', 0.2, 0.5, 0.75]
mape_results = []

for feat in max_features:
    rf = RandomForestRegressor(max_features=feat, random_state=50)
    rf.fit(x_train_yeezy, y_train_yeezy)
    valid_pred = rf.predict(x_valid_yeezy_rf)
    valid_error = valid_pred - y_valid_yeezy_rf
    mape = np.mean(100 * abs((valid_error/valid_pred)))
    mape_results.append(mape)

print(mape_results)


# In[103]:


# Lastly, we can tune our max features hyper parameter to determine the best MAPE value (off-white)
max_features = ['auto', 'sqrt', 0.2, 0.5, 0.75]
mape_results = []

for feat in max_features:
    rf = RandomForestRegressor(max_features=feat, random_state=50)
    rf.fit(x_train_ow, y_train_ow)
    valid_pred = rf.predict(x_valid_ow_rf)
    valid_error = valid_pred - y_valid_ow_rf
    mape = np.mean(100 * abs((valid_error/valid_pred)))
    mape_results.append(mape)

print(mape_results)


# ## Random Forest Model with Tuned Hyperparameters

# In[104]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Create model with our optimal hyperparams
rf_yeezy = RandomForestRegressor(n_estimators = 350, max_depth= 10, max_features='auto', random_state = 50)

# Fit the model on training data
rf_yeezy.fit(x_train_yeezy, y_train_yeezy)


# In[105]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Create model with our optimal hyperparams
rf_ow = RandomForestRegressor(n_estimators = 100, max_depth = 25, max_features='auto', random_state = 50)

# Fit the model on training data
rf_ow.fit(x_train_ow, y_train_ow)


# In[106]:


# Generating predictions and MAPE from our random forest on yeezy validation data
valid_pred = rf_yeezy.predict(x_valid_yeezy_rf)
valid_error = valid_pred - y_valid_yeezy_rf
valid_mape = np.mean(100 * abs((valid_error/valid_pred)))
valid_mape


# In[107]:


# Generating predictions and MAPE from our random forest on off-white validation data
valid_pred = rf.predict(x_valid_ow_rf)
valid_error = valid_pred - y_valid_ow_rf
valid_mape = np.mean(100 * abs((valid_error/valid_pred)))
valid_mape


# Based on our MAPEs from the linear regression models, we have obtained a smaller MAPE for both the yeezy and off-white data using a random forest model. So far, these are our best candidates.

# ### Get Feature Importance from Random Forest Models

# In[108]:


# Before we get the final pred on the test data we can get a sense of feature importance as well
feature_list = list(x_train_yeezy.columns)

# Get numerical feature importances
importances = list(rf_yeezy.feature_importances_)

# List of tuples with each variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {} Importance: {}'.format(*pair)) for pair in feature_importances]


# In[109]:


# Before we get final pred on the test data we can get a sense of feature importance as well
feature_list = list(x_train_ow.columns)

# Get numerical feature importances
importances = list(rf_ow.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {} Importance: {}'.format(*pair)) for pair in feature_importances]


# ### Rerun of our Random Forest Models with Subsetted Features

# In[110]:


# Create the model with our tuned hyperparams
rf_yeezy_important = RandomForestRegressor(n_estimators = 350, max_depth= 10, max_features='auto', random_state = 50)

# Create the dataframes with only our important features
x_train_yeezy_imp = x_train_yeezy.drop(columns=['Color_Blue', 'Release_Day_Tuesday', 'Release Month_August', 'Release Month_December', 'Release Month_February'])
x_valid_yeezy_rf_imp = x_valid_yeezy_rf.drop(columns=['Color_Blue', 'Release_Day_Tuesday', 'Release Month_August', 'Release Month_December', 'Release Month_February'])

# Fit the model on training data
rf_yeezy_important.fit(x_train_yeezy_imp, y_train_yeezy)


# In[111]:


# Generating predictions from our random forest with only important features on yeezy valid data
valid_pred = rf_yeezy_important.predict(x_valid_yeezy_rf_imp)
valid_error = valid_pred - y_valid_yeezy_rf
valid_mape = np.mean(100 * abs((valid_error/valid_pred)))
valid_mape


# In[112]:


# Create the model with our tuned hyperparams
rf_ow_important = RandomForestRegressor(n_estimators = 100, max_depth = 25, max_features='auto', random_state = 50)

# Create the dataframes with only our important features
x_train_ow_imp = x_train_ow.drop(columns=['Color_Black', 'Air_Max', 'Blazer', 'Air_Force', 'Release_Day_Friday', 'Release_Day_Saturday', 'Release_Day_Thursday', 'Release_Day_Wednesday', 'Release Month_April', 'Release Month_August', 'Release Month_February', 'Release Month_March', 'Release Month_November', 'Release Month_October'])
x_valid_ow_rf_imp = x_valid_ow_rf.drop(columns=['Color_Black', 'Air_Max', 'Blazer', 'Air_Force', 'Release_Day_Friday', 'Release_Day_Saturday', 'Release_Day_Thursday', 'Release_Day_Wednesday', 'Release Month_April', 'Release Month_August', 'Release Month_February', 'Release Month_March', 'Release Month_November', 'Release Month_October'])

# Fit the model on training data
rf_ow_important.fit(x_train_ow_imp, y_train_ow)


# In[113]:


# Generating predictions from our random forest with only important features on off-white valid data
valid_pred = rf_ow_important.predict(x_valid_ow_rf_imp)
valid_error = valid_pred - y_valid_ow_rf
valid_mape = np.mean(100 * abs((valid_error/valid_pred)))
valid_mape


# So far, our best MAPEs are still from the full random forest model with the tuned hyperparameters. Although the subsetted random forest model did not produce a better MAPE, we can still keep what features were important in mind.

# In[114]:


# Test MAPE for yeezy data
test_pred = rf_yeezy.predict(x_test_yeezy_rf)
test_error = test_pred - y_test_yeezy_rf
test_mape = np.mean(100 * abs((test_error/test_pred)))
test_mape


# In[115]:


# Test MAPE for off-white data
test_pred = rf_ow.predict(x_test_ow_rf)
test_error = test_pred - y_test_ow_rf
test_mape = np.mean(100 * abs((test_error/test_pred)))
test_mape


# ## Using our final Random Forest Models on newly obtained data

# Although our test data gives us an indication of how our model would perform on new data, I collected a few new observations based on yeezy shoes that had come out after our original data was recorded. I added in a column for the range of prices that the particular shoe sold for on StockX to see if our model would predict the price of each shoe in the actual range it sold for. There were not many off-white collabs similar to the data we originally had, so we will only be conducting this step for the yeezy model.

# In[116]:


# Set our working directory and pull in the new 
os.chdir("C:\\Users\\jorda\\OneDrive\\Documents\\IAA Files and Documents\\Side Project Files\\StockX")
yeezy_new = pd.read_excel("new_data.xlsx", sheet_name = 'Yeezy')


# In[117]:


# Dropping the columns that we do not need in the model
y_new = yeezy_new.drop(columns=['Shoe Name', 'Brand', 'Range of Prices'])


# In[118]:


# Get dummy values for our categorical variables
y_new = pd.get_dummies(y_new)


# In[119]:


# Fit our random forest model to all of our original data
rf_final_yeezy = RandomForestRegressor(n_estimators = 350, max_depth= 10, max_features='auto', random_state = 50)

# Fit the model on our original data (all combined)
rf_final_yeezy.fit(x_yeezy, y_yeezy)


# In[120]:


# Generate predictions on our new data
yeezy_new_pred = rf_final_yeezy.predict(y_new)


# In[121]:


# Output the models predicted values and the actual range of the observations from StockX
Assesment_Yeezy = pd.DataFrame({'Shoe Name':yeezy_new['Shoe Name'], 'Predicted':yeezy_new_pred, 'Actual Range':yeezy_new['Range of Prices']})
Assesment_Yeezy


# ## 4. Final Conclusions

# In conclusion, we found that our random forest models with all features included produced the lowest MAPE value for both the Yeezy and Off-White data. Even though they were lower, our model predictions are still about 14% off from the true value. By looking at our predicted vs actual range list above, we can see the model only predicted 5 out of the 16 observations in the correct range of sale prices. 
# 
# Further feature engineering may be necessary in order to help the model learn the differences in certain shoes, such as different colorways, how "hype" the shoe is before it releases, and reflective vs non reflective shoes. For example, looking at the table above we can see that the Black Reflective and Black Non Reflective were predicted to be the same price but the Non Reflective shoes went for way less than the Reflective shoes. By adding in extra features, it could help the model learn to differentiate between these type of shoes and give more accurate prices.
# 
# Some of the most important features, based on our random forest models, that influence sale price of shoes were the color black in the title of the shoe name and whether the shoe was a V2 or not for Yeezys and the color blue in the title of the shoe name and whether the shoe was a Jordan 1 or not for Off-Whites.
# 
# All in all, this was an extremely fun project to work on to test my skills and see if I could derive any results from an area of interest! This project could be further developed to produce even more accurate predictions but would require more data collection, that is hard to obtain from StockX, or a new target for the project could be used. The possibilities are endless!

# <b>Created by: Jordan Strickland, jmstric5@ncsu.edu<b>
