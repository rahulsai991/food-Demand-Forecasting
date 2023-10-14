#!/usr/bin/env python
# coding: utf-8

# # Food Demand Forecasting

# ## Python Libraries

# In[1]:


from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_csv('/kaggle/input/food-demand-forecasting/train.csv')
test = pd.read_csv('/kaggle/input/food-demand-forecasting/test.csv')
meal = pd.read_csv('/kaggle/input/food-demand-forecasting/meal_info.csv')
centerinfo = pd.read_csv('/kaggle/input/food-demand-forecasting/fulfilment_center_info.csv')


# In[3]:


train.head()


# In[4]:


centerinfo.head()


# In[5]:


meal.head()


# In[6]:


train.describe()


# In[7]:


train.info()


# ## Light Data Exploration
# ### 1) For numeric data
#   * Made histograms to understand distributions
#   * Corrplot
# 
# ### 2) For Categorical Data
#    * Made bar charts to understand balance of classes

# In[8]:


train_cat = train[['center_id','meal_id','emailer_for_promotion','homepage_featured']]
train_num = train[['week','checkout_price']]


# In[9]:


for i in train_num.columns:
    plt.hist(train_num[i])
    plt.title(i)
    plt.show()


# In[10]:


sns.heatmap(train_num.corr())


# In[11]:


for i in train_cat.columns:
    plt.xticks(rotation=90)
    sns.barplot(train_cat[i].value_counts().index,train_cat[i].value_counts()).set_title(i)
    plt.show()
    


# ## Data Normalization
# 1. for-loop: here we checked outliers occur or not? "checkout_price" column has occurred an outlier. 
# 2. outlinefree() : It is a customise function that help us to figureout and work on outlier values in columns. meanly, it is used to **remove outlires** values from dataset.
# 3. for-loop: with the help of for-loop, we are checking the **outlinefree()** function worked properly or not.
# 4. columns **center_id** and **meal_id** has many categorical values.
# 5. to manage categorical columns we using function their create new few sub-categories.
# 

# In[12]:


for i in train_num.columns:
    sns.boxplot(train_num[i])
    plt.title(i)
    plt.show()


# In[13]:


def outlinefree(dataCol):     
      
    sorted(dataCol)                          # sort column
    Q1,Q3 = np.percentile(dataCol,[25,75])   # getting 25% and 75% percentile
    IQR = Q3-Q1                              # getting IQR 
    LowerRange = Q1-(1.5 * IQR)              # getting Lowrange
    UpperRange = Q3+(1.5 * IQR)              # getting Upperrange 
    
    colname = dataCol.tolist()               # convert column into list  
    newlist =[]                              # empty list for store new values
    for i in range(len(colname)):
        
        if colname[i] > UpperRange:          # list number > Upperrange 
            colname[i] = UpperRange          # then number = Upperrange
            newlist.append(colname[i])       # append value to empty list
        elif colname[i] < LowerRange:        # list number < Lowrange 
            colname[i] = LowerRange          # then number = Lowrange
            newlist.append(colname[i])       # append value to empty list 
        else:
            colname[i]                       # list number
            newlist.append(colname[i])       # append value to empty list
            
        

    return newlist


# In[14]:


for i in range(len(train_num.columns)):
    new_list =  outlinefree(train.loc[:,train_num.columns[i]]) # retrun new list
    train.loc[:,train_num.columns[i]] = new_list 


# In[15]:


def center_id(datacol):
    center_id_val_index_n = []
    for i in datacol:
        if i >= 10 and i <= 30:
            center_id_val_index_n.append("10-30")
        elif i >= 31 and i <=50:
            center_id_val_index_n.append("31-50")
        elif i >= 51 and i <=70:
            center_id_val_index_n.append("51-70")  
        elif i >= 71 and i <=90:
            center_id_val_index_n.append("71-90")
        elif i >= 91 and i <=110:
            center_id_val_index_n.append("91-110") 
        elif i >= 111 and i <=130:
            center_id_val_index_n.append("111-130")
        elif i >= 131 and i <=150:
            center_id_val_index_n.append("131-150")          
        else:
            center_id_val_index_n.append("151-190")
    
    return  center_id_val_index_n 
center_id_val_index_n = center_id(train.center_id) 
train.center_id = center_id_val_index_n


# In[16]:


def meal_id(datacol):        
    meal_id_val_index_n = []
    for i in datacol:
        if i >= 1000 and i <= 1300:
            meal_id_val_index_n.append("1000-1300")
        elif i >= 1301 and i <=1600:
            meal_id_val_index_n.append("1301-1600")
        elif i >= 1601 and i <=1900:
            meal_id_val_index_n.append("1601-1900")  
        elif i >= 1901 and i <=2200:
            meal_id_val_index_n.append("1901-2200")
        elif i >= 2201 and i <=2500:
            meal_id_val_index_n.append("2201-2500") 
        elif i >= 2501 and i <=2800:
            meal_id_val_index_n.append("2501-2800")          
        else:
            meal_id_val_index_n.append("2801-3000") 
    return  meal_id_val_index_n

meal_id_val_index_n = meal_id(train.meal_id)
train.meal_id = meal_id_val_index_n


# ## Feature Selection
# 1. seaborn.pairplot(): It is help to figure-out relation between features and label.

# In[17]:


sns.pairplot(train)


# In[18]:


f_train = train.loc[:,['num_orders','week','center_id','meal_id','checkout_price','base_price','emailer_for_promotion',
                 'homepage_featured']]
final_train = pd.get_dummies(f_train)


# In[19]:


features = final_train.iloc[:,1:].values
label = final_train.iloc[:,:1].values


# ## Model Buliding
# here we will be using many algorithms and compare all of them. which algorithm will be giving us a Better result. The following algorithms are below.
# 
# 1. LinearRegression (RMSE: 334.45162241353864)
# 2. DecisionTreeRegressor (RMSE:  332.8261160204239)
# 3. **RandomForestRegressor (RMSE: 331.0142032987282)**

# In[20]:


#------------------------------------ LinearRegression ---------------------------------------------
X_train,X_test,y_train,y_test = train_test_split(features,label,test_size=0.20,random_state=1705)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# In[21]:


print("R2 score  :",r2_score(y_test, y_pred))
print("MSE score  :",mean_squared_error(y_test, y_pred))
print("RMSE: ",sqrt(mean_squared_error(y_test, y_pred)))


# In[22]:


#------------------------------------ DecisionTreeRegressor---------------------------------------------
X_train,X_test,y_train,y_test = train_test_split(features,label,test_size=0.20,random_state=1956)
DTRmodel = DecisionTreeRegressor(max_depth=3,random_state=0)
DTRmodel.fit(X_train,y_train)
y_pred = DTRmodel.predict(X_test)


# In[23]:


print("R2 score  :",r2_score(y_test, y_pred))
print("MSE score  :",mean_squared_error(y_test, y_pred))
print("RMSE: ",sqrt(mean_squared_error(y_test, y_pred)))


# In[24]:


#------------------------------------ RandomForestRegressor ---------------------------------------------
X_train,X_test,y_train,y_test = train_test_split(features,label,test_size=0.20,random_state=33)
RFRmodel = RandomForestRegressor(max_depth=3, random_state=0)
RFRmodel.fit(X_train,y_train)
y_pred = RFRmodel.predict(X_test)


# In[25]:


print("R2 score  :",r2_score(y_test, y_pred))
print("MSE score  :",mean_squared_error(y_test, y_pred))
print("RMSE: ",sqrt(mean_squared_error(y_test, y_pred)))


# ## Conclusion
# I will choose a **RandomForestRegressor algorithm** for this dataset.
# 
# **RandomForestRegressor score**:
# 
# 1. **RMSE score : 331.0142032987282** 
# 

# ## Algorithm applied
# before applying the algorithm to the test dataset. we should make it a complete numeric dataset. the following setups are below mentioned.
# 1. columns center_id and meal_id has many categorical values.
# 2. to manage categorical columns we using function their create new few sub-categories.
# 3. using get_dummies() function.
# 4. here our data is ready to apply an algorithm on it.

# In[26]:


center_id_val_index_n = center_id(test.center_id) 
test.center_id = center_id_val_index_n

meal_id_val_index_n = meal_id(test.meal_id)
test.meal_id = meal_id_val_index_n


# In[27]:


f_test = test.loc[:,['week','center_id','meal_id','checkout_price','base_price','emailer_for_promotion',
                 'homepage_featured']]
final_test = pd.get_dummies(f_test)


# In[28]:


test_predict = RFRmodel.predict(final_test)


# In[29]:


test['num_orders'] = test_predict


# In[30]:


sample =  test.loc[:,['id','num_orders']]


# In[31]:


sample.to_csv('sample_submission.csv',index=False)

