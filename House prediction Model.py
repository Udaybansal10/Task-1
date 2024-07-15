#!/usr/bin/env python
# coding: utf-8

# # Importing Required Libraries

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # Analysing Dataset

# In[28]:


df = pd.read_csv('C:/Users/udayb/Downloads/house-prices-advanced-regression-techniques/train.csv')
df.head()


# In[5]:


df.info()


# # Data Cleaning

# In[6]:


elements_to_keep = ['Id', 'LotArea', 'FullBath', 'BedroomAbvGr', 'SalePrice']
df = df[elements_to_keep]
df.head()


# # Missing for missing/null values

# In[7]:


print(df.isnull().sum())


# # Checking Duplicated values

# In[31]:


print("Number of duplicate values:", df.duplicated().sum())


# In[9]:


plt.figure(figsize=(16,5))
plt.subplot(2,2,1)
sns.histplot(df['LotArea'])

plt.subplot(2,2,2)
sns.histplot(df['FullBath'])

plt.subplot(2,2,3)
sns.histplot(df['BedroomAbvGr'])

plt.subplot(2,2,4)
sns.histplot(df['SalePrice'])

plt.tight_layout()
plt.show()


# # checking for outlier if any

# In[10]:


plt.figure(figsize=(16,5))
plt.subplot(3,1,1, facecolor = 'red')
sns.boxplot(x = df['LotArea'], width = 0.3)

plt.subplot(3,1,2, facecolor = 'lightblue')
sns.boxplot(x = df['FullBath'], width = 0.3)

plt.subplot(3,1,3, facecolor = 'yellow')
sns.boxplot(x = df['BedroomAbvGr'], width = 0.3)

plt.tight_layout()
plt.show()


# # Removing outlier in LotArea

# In[11]:


percentile25 = df['LotArea'].quantile(0.25)
percentile75 = df['LotArea'].quantile(0.75)

iqr = percentile75 - percentile25

upperlimit = percentile75 + 1.5 * iqr
lowerlimit = percentile25 - 1.5 * iqr

print("LotArea")
print("UpperLimit", upperlimit)
print("LowerLimit", lowerlimit)

new_df = df.copy()
new_df['LotArea'] = np.where(new_df['LotArea'] > upperlimit, upperlimit, np.where(new_df['LotArea'] < lowerlimit, lowerlimit, new_df['LotArea']))


# plt.figure(figsize=(16,5))
# plt.subplot(3,1,1, facecolor = 'red')
# sns.boxplot(x = df['LotArea'], width = 0.3)
# 
# plt.subplot(3,1,2, facecolor = 'red')
# sns.boxplot(x = new_df['LotArea'], width = 0.3)
# 
# 
# plt.tight_layout()
# plt.show()

# # Removing outlier in No. of Bathrooms

# In[12]:


percentile25 = df['FullBath'].quantile(0.25)
percentile75 = df['FullBath'].quantile(0.75)

iqr = percentile75 - percentile25

upperlimit = percentile75 + 1.5 * iqr
lowerlimit = percentile25 - 1.5 * iqr

print("LotArea")
print("UpperLimit", upperlimit)
print("LowerLimit", lowerlimit)

new_df1 = df.copy()
new_df1['FullBath'] = np.where(new_df1['FullBath'] > upperlimit, upperlimit, np.where(new_df1['FullBath'] < lowerlimit, lowerlimit, new_df1['FullBath']))


# In[13]:


plt.figure(figsize=(16,5))
plt.subplot(3,1,1, facecolor = 'lightblue')
sns.boxplot(x = df['FullBath'], width = 0.3)

plt.subplot(3,1,2, facecolor = 'lightblue')
sns.boxplot(x = new_df1['FullBath'], width = 0.3)


plt.tight_layout()
plt.show()


# # Removing outlier in No. of Bedrooms

# In[14]:


percentile25 = df['BedroomAbvGr'].quantile(0.25)
percentile75 = df['BedroomAbvGr'].quantile(0.75)

iqr = percentile75 - percentile25

upperlimit = percentile75 + 1.5 * iqr
lowerlimit = percentile25 - 1.5 * iqr

print("LotArea")
print("UpperLimit", upperlimit)
print("LowerLimit", lowerlimit)

new_df2 = df.copy()
new_df2['BedroomAbvGr'] = np.where(new_df2['BedroomAbvGr'] > upperlimit, upperlimit, np.where(new_df2['BedroomAbvGr'] < lowerlimit, lowerlimit, new_df2['BedroomAbvGr']))


# plt.figure(figsize=(16,5))
# 
# plt.subplot(3,1,1, facecolor = 'yellow')
# sns.boxplot(x = df['BedroomAbvGr'], width = 0.3)
# 
# plt.subplot(3,1,2, facecolor = 'yellow')
# sns.boxplot(x = new_df2['BedroomAbvGr'], width = 0.3)
# 
# plt.tight_layout()
# plt.show()

# # Applying ML Algorithm - Linear Regression

# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[16]:


X = df.iloc[:,1:4]  # Selecting columns 1, 2, 3 as features
y = df.iloc[:,-1]  # Assuming column 4 is your target variable


# In[17]:


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


# Initialize and train machine learning model
model = LinearRegression()
model.fit(X_train, y_train)


# In[19]:


# Make predictions
y_pred = model.predict(X_test)


# # Calculating cross validation score for liner model

# In[20]:


# Perform cross-validation on the training set
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,X_train, y_train, cv=5, scoring='neg_mean_squared_error')
mse_scores = -scores 
mean_cv_mse = np.mean(mse_scores)*100
print(f'Cross Validated Score: {mean_cv_mse}')


# # Calculating Mean Squared error, Mean Absolute error and R2 score

# In[21]:


from sklearn.metrics import  mean_absolute_error, mean_squared_error, r2_score


# In[22]:


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print("Mean Absolute Error:", mae)
print(f'R^2 Score: {r2}')


# # Scatter Plot for Actual V/s Predicted Values

# In[23]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Sale Prices')
plt.show()


# # Prediction of Random values

# In[24]:


features = ['LotArea', 'FullBath', 'BedroomAbvGr']
target = 'SalePrice'


# In[25]:


random_test_cases = df.sample(5, random_state=42)
random_test_features = random_test_cases[features]
random_test_actual_prices = random_test_cases[target]

random_predictions = model.predict(random_test_features)

comparison_df = random_test_cases.copy()
comparison_df['PredictedSalePrice'] = random_predictions

print("\nRandom Test Cases with Actual and Predicted Sale Prices:\n", comparison_df[['LotArea', 'FullBath', 'BedroomAbvGr', 'SalePrice', 'PredictedSalePrice']])

