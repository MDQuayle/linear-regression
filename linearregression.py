import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()


# ## Loading the raw data

# In[3]:


raw_data = pd.read_csv('1.04. Real-life example.csv')
raw_data.head()


# ## Preprocessing

# ### Exploring the descriptive statistics of the variables

# In[4]:


raw_data.describe(include='all')


# ### Determining the variables of interest

# In[5]:


data = raw_data.drop(['Model'],axis=1)
data.describe(include='all')


# ### Dealing with missing values

# In[6]:


data.isnull().sum()


# In[7]:


data_no_mv = data.dropna(axis=0)


# In[8]:


data_no_mv.describe(include='all')


# ### Exploring the PDFs

# In[9]:


sns.distplot(data_no_mv['Price'])


# ### Dealing with outliers

# In[10]:


q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
data_1.describe(include='all')


# In[11]:


sns.distplot(data_1['Price'])


# In[12]:


sns.distplot(data_no_mv['Mileage'])


# In[13]:


q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]


# In[14]:


sns.distplot(data_2['Mileage'])


# In[15]:


sns.distplot(data_no_mv['EngineV'])


# In[16]:


data_3 = data_2[data_2['EngineV']<6.5]


# In[17]:


sns.distplot(data_3['EngineV'])


# In[18]:


sns.distplot(data_no_mv['Year'])


# In[19]:


q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]


# In[20]:


sns.distplot(data_4['Year'])


# In[21]:


data_cleaned = data_4.reset_index(drop=True)


# In[22]:


data_cleaned.describe(include='all')


# ## Checking the OLS assumptions

# In[23]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')


plt.show()


# In[24]:


sns.distplot(data_cleaned['Price'])


# ### Relaxing the assumptions

# In[25]:


log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
data_cleaned


# In[26]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')


plt.show()


# In[27]:


data_cleaned = data_cleaned.drop(['Price'],axis=1)


# ### Multicollinearity

# In[28]:


data_cleaned.columns.values


# In[29]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns


# In[30]:


vif


# In[31]:


data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)
