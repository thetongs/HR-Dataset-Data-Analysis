#!/usr/bin/env python
# coding: utf-8

# In[2]:


## Data analysis and Data preprocessing on HR dataset
#


# In[98]:


## Load libraries
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[119]:


## Load dataset
# seperate dataset
dataset1 = pd.read_csv('aug_train.csv')
dataset2 = pd.read_csv('aug_test.csv')

# for analysis combine those dataset into one vertically
dataset = dataset1.append(dataset2, ignore_index = True)
column_name = dataset.columns
dataset.head()


# In[120]:


## General information about dataset
# number of records
print("Total records are : {}".format(len(dataset)))
print("Total records are : {}".format(dataset.shape[0]))


# In[121]:


# total number of columns
print("Total columns are : {}".format(len(dataset.columns)))
print("Total columns are : {}".format(dataset.shape[1]))


# In[122]:


# Records and Columns together 
print("Shape of dataset : {}".format(dataset.shape))


# In[123]:


## General information
#
dataset.info()


# In[124]:


## Statistical Information
#
dataset.describe()


# In[125]:


## Check for missing values of each column
#
dataset.isna().sum()


# In[126]:


# in percentange
NAN = [(clm_name, dataset[clm_name].isna().mean()*100) for clm_name in dataset]
NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])
NAN


# In[127]:


# set one threshold like 50% 
# so if column has more than 50 % data missing just drop that column
NAN[NAN['percentage'] > 50]


# In[128]:


## Check data types and handle it
dataset.dtypes


# In[129]:


# change if required
# using dictionary to convert specific columns 
convert_dict = {'city': 'category',
                'gender': 'category',
                'relevent_experience':'category',
                'enrolled_university':'category',
                'education_level':'category',
                'major_discipline':
                'category',
                'experience':'category',
                'company_size':'category',
                'company_type':'category',
                'last_new_job': 'category'
               } 
  
dataset = dataset.astype(convert_dict) 
dataset.dtypes


# In[130]:


# Using SimpleImputer
# After using simple imputer the column names 
# change into numeric index
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="most_frequent")
dataset = imputer.fit_transform(dataset)

dataset = pd.DataFrame(dataset)
dataset.columns = column_name
dataset.isna().sum()


# In[131]:


dataset.head()


# In[132]:


## Dataset with original labels
dataset3 = dataset.copy()
dataset3.head()


# In[133]:


## Handle categorical values
# using label encoder for multiple columns
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

cl_names = ['city','gender','relevent_experience','enrolled_university',
 'education_level',
 'major_discipline',
 'experience',
 'company_size',
 'company_type',
 'last_new_job']

dataset[cl_names] = dataset[cl_names].apply(encoder.fit_transform)
dataset.head()


# In[134]:


## To keep track of encoding label
#
encoder.fit_transform(dataset3['gender'])


# In[140]:


# back to original
encoder.inverse_transform([[0]])


# In[136]:


## Number of males and females
# 0 - female , 1- male 2- other 
dataset.gender.value_counts()


# In[151]:


## Visualize it
#
names = ['male', 'female', 'others']
cat = dataset.gender.value_counts()
cat = list(cat)

plt.bar(names, cat)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Number of genders of each category')
plt.show()
plt.close()


# In[152]:


dataset.target.value_counts()


# In[153]:


## Number of employees looking for new job and others
#
## Visualize it
#
names = ['not looking for new job', 'looking for new job']
cat = dataset.target.value_counts()
cat = list(cat)

plt.bar(names, cat)
plt.xlabel('type of employee')
plt.ylabel('count')
plt.title('category of employee')
plt.show()
plt.close()


# In[161]:


import plotly.express as px
px.box(data_frame=dataset,x='target',y='city_development_index')


# In[165]:


import plotly.express as px
px.box(data_frame=dataset,x='target',y='relevent_experience')


# In[167]:


## Calculate measure of central dependancy
# Mean
print("The average value of each columns are below.")
print("Mean\n{}\n".format(dataset.mean()))


# In[168]:


# Median
print("The middle value of all the columns are below.")
print("Median\n{}\n".format(dataset.median()))


# In[169]:


# Mode
print("The most common value of each column are below.")
print("Mode\n{}\n".format(dataset.mode().iloc[0]))


# In[171]:


## Measure of dispersion
# Varience
print("Variance\n{}\n".format(dataset.var()))
print("""
If variance is small
- It means all column datapoints are tend to close together and close to mean.
If variance is big
- It means this column datapoints are spread-out with respect to each other and with respect to mean.
""")


# In[172]:


# Standard deviation
print("Standard deviation\n{}\n".format(dataset.std()))
print("""
Standard deviation is small.
- It means data points are tightky clustered around mean.
Standard deviation is big.
- It means data points widely spread as compare to other columns.
""")


# In[173]:


## Calculate moments
#
from scipy.stats import kurtosis
from scipy.stats import skew
# Skewness
print("Skewness\n{}\n".format(dataset.skew()))
skews = dataset.skew()
sk_list = list()

for i in skews:
    if(i == 0):
        sk_list.append("Normally distributed")
    elif(i < 0):
        sk_list.append("Negatively distributed")
    elif(i>0):
        sk_list.append("Positively distributed")
skewness_result = pd.Series(sk_list)
skewness_result.index = dataset.mean().index

print("The details informaton about skewness below.")
print(skewness_result)


# In[174]:


# Kurtosis
print("Kurtosis\n{}\n".format(dataset.kurtosis()))
kur = dataset.kurtosis()
sk_list = list()
for i in kur:
    if(i == 0):
        sk_list.append("Mesokurtic")
    elif(i < 0):
        sk_list.append("Leptokurtic")
    elif(i>0):
        sk_list.append("Platykurtic")
kurtosis_result = pd.Series(sk_list)
kurtosis_result.index = dataset.mean().index

print("The details informaton about kurtosis below.")
print(kurtosis_result)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


##

