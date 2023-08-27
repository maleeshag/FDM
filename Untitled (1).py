#!/usr/bin/env python
# coding: utf-8

# In[138]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier,plot_tree,DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report


# In[109]:


df=pd.read_csv("C:\\Users\\it21178368\\Desktop\\Practical 04-weather dataset.csv")


# In[110]:


df.head()


# In[44]:


df.drop("Unnamed: 0",axis=1)


# In[45]:


df.head()


# In[46]:


df.drop("Unnamed: 0",axis=1,inplace=True)


# In[47]:


df.head()


# In[49]:


df.tail(3)


# In[51]:


le=LabelEncoder()


# In[54]:


df["Play golf"]=le.fit_transform(df["Play golf"])       


# In[55]:


df.head()


# In[64]:


data=df.apply(le.fit_transform)


# In[66]:


data.head()


# In[67]:


data.iloc[2,1]


# In[69]:


data.iloc[1,1:4]


# In[70]:


data.iloc[1:4,1:4]


# In[71]:


data.shape


# In[72]:


data.iloc[0:14,0:4]


# In[74]:


data.iloc[:,0:4]


# In[76]:


x=data.iloc[:,:4].values


# In[77]:


y=data.iloc[:,4].values


# In[85]:


train_test_split(x,y,test_size=0.2,random_state=1)


# In[95]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[96]:


x_train


# In[102]:


model=DecisionTreeClassifier(criterion="gini")


# In[103]:


model.fit(x_train,y_train)


# In[127]:


plt.figure(figsize=(15,15))
plot_tree(model,feature_names=data.columns,class_names=["Yes","No"],filled=True)
plt.show


# In[119]:


df.columns


# In[134]:


y_pred=model.predict(x_test)


# In[135]:


accuracy_score(y_test,y_pred)


# In[136]:


model.predict([[1,0,0,1]])


# In[ ]:




