#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm
from scipy.stats import t
from numpy.random import seed


# # Load Test and Train Files

# In[2]:


file1 = "/Users/ayadav/Downloads/164410_790308_bundle_archive/test.csv"


# In[3]:


file2 = "/Users/ayadav/Downloads/164410_790308_bundle_archive/train.csv"


# In[4]:


df_train = pd.read_csv(file2)


# In[5]:


df_test = pd.read_csv(file1)


# # Check No.of Columns & Rows

# In[6]:


df_train.shape


# In[29]:


df_test.shape


# In[8]:


df_train.head()


# # Work with Train Data

# In[ ]:





# # Exploration of Data

# In[30]:


df_train.head()


# In[31]:


df_train.describe()


# In[ ]:





# #Missing Values

# In[33]:


missing_var= df_train.isnull().sum()
print(missing_var)


# In[14]:


df = df.sort_index(axis=1)


# In[15]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# In[35]:


df_train.select_dtypes(include=['object']).head()


# In[36]:


df_train.describe()


# In[38]:


# # Missing Values

# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[39]:


missing_values_table(df_train).head(50)


# In[43]:


# # # Exploratory Data Analysis

for i in df_train.select_dtypes(include=['object']).columns:
    df_train.drop(labels=i, axis=1, inplace=True)


# In[74]:


data_to_plot = ['PRI_SANCTIONED_AMOUNT']
       


# In[75]:


df_train.columns


# In[79]:


# Create a figure instance
fig = plt.figure(1, figsize=(4, 20))

# Create an axes instance
#ax = fig.add_subplot()

# Create the boxplot
df_train.boxplot(data_to_plot)
plt.show()


# In[ ]:


'VOTERID_FLAG', 'DRIVING_FLAG', 'PASSPORT_FLAG', 'PERFORM_CNS_SCORE',
       'PRI_NO_OF_ACCTS', 'PRI_ACTIVE_ACCTS', 'PRI_OVERDUE_ACCTS',
       'PRI_CURRENT_BALANCE', 'PRI_SANCTIONED_AMOUNT', 'PRI_DISBURSED_AMOUNT',
       'SEC_NO_OF_ACCTS', 'SEC_ACTIVE_ACCTS', 'SEC_OVERDUE_ACCTS',
       'SEC_CURRENT_BALANCE', 'SEC_SANCTIONED_AMOUNT', 'SEC_DISBURSED_AMOUNT',
       'PRIMARY_INSTAL_AMT', 'SEC_INSTAL_AMT', 'NEW_ACCTS_IN_LAST_SIX_MONTHS',
       'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS', 'NO_OF_INQUIRIES',]

