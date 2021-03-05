#!/usr/bin/env python
# coding: utf-8

# ## Name - Kanigolla Likhita
# 
# ## Task 1 - Prediction using Supervised ML
# 
# ##  Predict the percentage of an student based on the no. of study hours.
# 
# ## Simple Linear Regression with two variables

# ######  In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# ### Importing Libraries and dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


url= "http://bit.ly/w-data"
data=pd.read_csv(url)
data.head(10)


# In[13]:


data.shape


# In[12]:


data.describe().T


# ### Plotting the distribution of scores

# In[3]:


data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel("No:of:Hours Studied")
plt.ylabel('Percentage Score')
plt.show()


# ### Preparing the data

# In[4]:


X=data.iloc[:,:-1].values
Y=data.iloc[:,1].values


# ### Splitting the data for Training and Testing

# In[5]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# ### Training the Algorithm

# In[6]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
print("Coefficients :", lr.coef_)
print("Intercepts :",lr.intercept_)


# ### Plotting the regression line

# In[7]:


line = lr.coef_* X + lr.intercept_
plt.scatter(X,Y)
plt.plot(X,line)
plt.show()


# ### Making Predictions

# In[8]:


print(X_test)
Y_pred=lr.predict(X_test)


# ### Actual vs predicted 

# In[9]:


df=pd.DataFrame({'Actual':Y_test,'Predicted':Y_pred})
df.head(5)


# ## Task: What will be predicted score if a student studies for 9.25 hrs/ day?
# 

# In[10]:


hours=9.25
hour=np.array(hours).reshape(-1,1)
prediction=lr.predict(hour)
print("No:of hours={} ".format(hour[0]))
print("Predicted Score = {}".format(prediction[0]))


# ### Evaluating model

# In[11]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred)) 
print('Mean squared Error:', metrics.mean_squared_error(Y_test, Y_pred)) 

