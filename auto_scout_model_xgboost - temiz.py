#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


# In[2]:


df_org = pd.read_csv("final_scout_not_dummy.csv")


# In[3]:


df_org.head(1)


# In[4]:


df = pd.concat([df_org["make_model"],df_org["body_type"], df_org["km"], df_org["age"], df_org["Gearing_Type"],df_org["Fuel"],df_org["hp_kW"], df_org["price"]], axis=1)
df


# In[5]:


df.to_csv("final_model.csv", index=False)


# In[6]:


df.info()


# # Train test split

# In[7]:


X=df.drop("price", axis=1)
y=df.price


# In[8]:


cat = X.select_dtypes("object").columns
cat


# In[9]:


from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
X[cat] = enc.fit_transform(X[cat])
X.head()


# In[10]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

print("Train features shape : ", X_train.shape)
print("Train target shape   : ", y_train.shape)
print("Test features shape  : ", X_test.shape)
print("Test target shape    : ", y_test.shape)


# # Modeling

# In[11]:


import xgboost


# In[12]:


from xgboost import XGBRegressor


# In[13]:


xgb_model = XGBRegressor(random_state=101, objective="reg:squarederror")


# In[14]:


xgb_model.fit(X_train, y_train)


# In[15]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[16]:


def train_val(model, X_train, y_train, X_test, y_test):
    
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    scores = {"train": {"R2" : r2_score(y_train, y_train_pred),
    "mae" : mean_absolute_error(y_train, y_train_pred),
    "mse" : mean_squared_error(y_train, y_train_pred),                          
    "rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},
    
    "test": {"R2" : r2_score(y_test, y_pred),
    "mae" : mean_absolute_error(y_test, y_pred),
    "mse" : mean_squared_error(y_test, y_pred),
    "rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}}
    
    return pd.DataFrame(scores)


# In[17]:


train_val(xgb_model, X_train, y_train, X_test, y_test)


# # Final Model and Model Deployment

# In[18]:


import pickle


# In[19]:


final_model = XGBRegressor(objective="reg:squarederror").fit(X_train,y_train)


# In[20]:


pickle.dump(final_model, open("auto_scout.pkl", 'wb'))


# In[21]:


pickle.dump(enc, open("autoscout_encoder.pkl", 'wb'))


# In[ ]:




