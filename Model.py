#!/usr/bin/env python
# coding: utf-8


# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# In[2]:


df = pd.read_csv('hiring.csv')
df['experience'].fillna(0, inplace=True)
df['test_score'].fillna(df['test_score'].mean(), inplace=True)
X = df.iloc[:, :3]


# In[3]:


df


# In[4]:


#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]


# In[5]:


X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))


# In[6]:


X


# In[8]:


y = df.iloc[:, -1]


# In[9]:


y


# In[10]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[11]:


#Fitting model with trainig data
regressor.fit(X, y)


# In[12]:


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))


# In[13]:


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))


# In[ ]:




