#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statistics as st


# In[2]:


fPath = "diabetes_binary_health_indicators_BRFSS2015 data.csv"

data = pd.read_csv(fPath)


# In[17]:


def labelchecker(data):
    
    cols = data.columns
    
    for col in cols:
        nlabType = data[col].nunique()
        print(f"Number of unique labels in {col} is  : {nlabType}")
        print(data[col]. unique())
        print('                                      ')
        print('######################################')
        print('                                      ')


# In[18]:


labelchecker(data)


# In[19]:


data.shape


# In[20]:


data.duplicated().sum()


# In[21]:


data.isna().sum()


# In[22]:


data.info()


# In[25]:


data1 = data.drop_duplicates(keep='first')


# In[26]:


data1.duplicated().sum()


# In[28]:


data1.shape


# In[29]:


data1.info()


# In[30]:


data1.isna().sum()


# In[31]:


data1.columns


# In[34]:


data1.sample(25)


# In[38]:


plt.figure(figsize=(3,2))

sns.countplot(data1['Diabetes_binary'])


# In[40]:


data1.Diabetes_binary.value_counts()


# In[ ]:




