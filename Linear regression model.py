#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression
# 

# ## US Housing Dataset

# In[69]:


#importing necessary libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# In[36]:


#importing data set
df = pd.read_csv(r"C:\Users\VAGDEVI\downloads\USA_Housing.csv")
df.head()


# In[35]:


#independent variables
X = df.drop(['Price','Address'],axis=1).values
#dependent variable
Y = df['Price'].values


# ### Splitting the dataset into training set and testing set

# In[103]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size = 0.2, random_state = 0)


# ### Fitting simple linear regression to the training set

# In[104]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train, Y_train)


# ### predicting the test set

# In[105]:


y_pred = LR.predict(X_test)
print(y_pred)


# In[106]:


LR.predict([[79545.458574,5.682861,7.009188,4.09,23086.800503]])


# ### Evaluating the model

# In[107]:


from sklearn.metrics import r2_score
r2_score(Y_test,y_pred)


# ### Plotting the results

# In[85]:


plt.scatter(Y_test, y_pred, color = 'b')
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title('Actual VS Predicted values')


# ### Predicted values and differences

# In[97]:


pred = pd.DataFrame({'Actual value':Y_test, 'Predicted Value': y_pred, 'Difference': Y_test-y_pred})
pred


# In[ ]:




