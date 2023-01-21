#!/usr/bin/env python
# coding: utf-8

# In[1344]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import warnings 
warnings.filterwarnings ('ignore')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Read the Data from the Given excel file
# In[1339]:


get_ipython().system('pip install xgboost')


# In[1265]:


data_test=pd.read_csv('Consumer_Complaints_test.csv')


# In[1266]:


data_test.shape


# In[1267]:


data_test.head(2)


# In[1268]:


data_train=pd.read_csv('Consumer_Complaints_train.csv')
data_train.shape


# In[1269]:


data_train.head(2)

Check the data type for both data (test file and train file)
# In[1270]:


data_test.info()


# In[1271]:


data_train.info()

Do missing value analysis and drop columns where more than 25% of data are missing
# In[1272]:


round(data_test.isna().sum() / len(data_test)*100)
#Sub-product,Sub-issue,Consumer complaint narrative,Company public response,Tags,Consumer consent provided? must be deleted


# In[1273]:


round(data_train.isna().sum() / len(data_train)*100)
#Sub-product,Consumer complaint narrative,Sub-issue,Company public response,Tags,Consumer consent provided? must be deleted


# In[1274]:


def drop_cols_na(data, threshold):
    return data[data.columns[round(data.isna().sum() / len(data)*100) < threshold]]
cust_test=drop_cols_na(data_test,25)
cust_test.head(2)


# In[1275]:


cust_train=drop_cols_na(data_train,25)
cust_train.head(2)

Extracting Day, Month, and Year from Date Received Column and create new fields for a month, year, and day 
# In[1276]:


cust_test['Year']=pd.DatetimeIndex(cust_test['Date received']).year
cust_test['Month']=pd.DatetimeIndex(cust_test['Date received']).month
cust_test['Day']=pd.DatetimeIndex(cust_test['Date received']).day


# In[1277]:


cust_train['Year']=pd.DatetimeIndex(cust_train['Date received']).year
cust_train['Month']=pd.DatetimeIndex(cust_train['Date received']).month
cust_train['Day']=pd.DatetimeIndex(cust_train['Date received']).day
cust_train.head(2)

Calculate the Number of Days the Complaint was with the Company and create a new field as “Days held”
# In[1278]:


dtest_received=cust_test['Date received'].astype('datetime64[ns]')
cust_test['Date sent to company']=cust_test['Date sent to company'].astype('datetime64[ns]')

cust_test['Days held']=(cust_test['Date sent to company']-dtest_received).dt.days
cust_test.head(2)


# In[1279]:


dtrain_received=cust_train['Date received'].astype('datetime64[ns]')
cust_train['Date sent to company']=cust_train['Date sent to company'].astype('datetime64[ns]')

cust_train['Days held']=(cust_train['Date sent to company']-dtrain_received).dt.days
cust_train.head(2)

Drop "Date Received","Date Sent to Company","ZIP Code", "Complaint ID" fields
# In[1280]:


cust_test=cust_test.drop(['Date sent to company','Date received',"ZIP code","Complaint ID"],axis=1)
cust_test.head(2)


# In[1281]:


cust_train=cust_train.drop(['Date sent to company','Date received',"ZIP code","Complaint ID"],axis=1)
cust_train.head(2)

Imputing Null value in “State” by Mode
# In[1282]:


cust_test['State']=cust_test['State'].fillna(cust_test['State'].mode()[0])
cust_test['State'].isnull().sum()


# In[1283]:


cust_train['State']=cust_train['State'].fillna(cust_train['State'].mode()[0])
cust_train['State'].isnull().sum()

with the help of the days we calculated above, create a new field 'Week_Received' 
where we calculate the week based on the day of receiving.
# In[1284]:


cust_test['Week_Received']=dtest_received.dt.week
cust_test.head(2)


# In[1285]:


cust_train['Week_Received']=dtrain_received.dt.week
cust_train.head(2)

store data of disputed people into the “disputed_cons” variable for future tasks 
# In[1286]:


disputed_cons=cust_train[cust_train['Consumer disputed?']=='Yes']

Plot bar graph of the total no of disputes of consumers with the help of seaborn
# In[1287]:


plt.figure(figsize=(8,5))
plt.title("the total no of disputes of consumers")
sns.countplot(x=cust_train['Consumer disputed?'])
plt.show()


# Plot bar graph of the total no of disputes products-wise with the help of seaborn

# In[1288]:


plt.figure(figsize=(10,5))
plt.title("the total no of disputes products-wise")
sns.countplot(data=disputed_cons,y='Product')
plt.show()


# Plot bar graph of the total no of disputes with Top Issues by Highest Disputes, with the help of seaborn

# In[1289]:


plt.figure(figsize=(10,5))
plt.title("the total no of disputes with Top Issues by Highest Disputes")
sns.countplot(data=disputed_cons,y='Issue',order=disputed_cons['Issue'].value_counts().iloc[:8].index)
plt.show()

Plot bar graph of the total no of disputes by State with Maximum Disputes
# In[1290]:


plt.figure(figsize=(10,5))
plt.title("the total no of disputes by State with Maximum Disputes")
sns.countplot(data=disputed_cons,y='State',order=disputed_cons['State'].value_counts().iloc[:8].index)
plt.show()

Plot bar graph of the total no of disputes Submitted Via different source
# In[1291]:


plt.figure(figsize=(10,5))
plt.title("the total no of disputes Submitted Via different source")
sns.countplot(data=disputed_cons,y='Submitted via')
plt.show()

Plot bar graph of the total no of disputes where the Company's Response to the Complaints
# In[1292]:


plt.figure(figsize=(10,5))
plt.title("the total no of disputes where Company's Response to the Complaints")
sns.countplot(data=disputed_cons,y='Company response to consumer')
plt.show()

Plot bar graph of the total no of disputes where the Company's Response Leads to Disputes
# In[1293]:


plt.figure(figsize=(10,5))
plt.title("the total no of disputes where Company's Response to the Complaints")
sns.countplot(data=cust_train,y='Company response to consumer',hue='Consumer disputed?')
plt.show()

Plot bar graph of the total no of disputes. Whether there are Disputes Instead of Timely Response
# In[1294]:


plt.figure(figsize=(8,5))
plt.title("Disputes Instead of Timely Response")
sns.countplot(data=disputed_cons,x='Timely response?')
plt.show()

Plot bar graph of the total no of disputes over Year Wise Complaints
# In[1295]:


plt.figure(figsize=(8,5))
plt.title("total no of disputes over Year Wise Complaints")
sns.countplot(data=cust_train,x='Year',hue='Consumer disputed?')
plt.show()

Plot bar graph of the total no of disputes over Year Wise Disputes
# In[1296]:


plt.figure(figsize=(8,5))
plt.title("total no of disputes over Year Wise Disputes")
sns.countplot(data=disputed_cons,x='Year')
plt.show()

Plot bar graph of Top Companies with Highest Complaints
# In[1297]:


plt.figure(figsize=(10,5))
plt.title("Top Companies with Highest Complaints")
sns.countplot(data=disputed_cons,y='Company',order=disputed_cons['Company'].value_counts().iloc[:8].index)
plt.show()

Converte all negative days held to zero (it is the time taken by the authority that can't be negative)
# In[1298]:


np.min(cust_test['Days held']),np.min(cust_train['Days held'])


# In[1299]:


cust_test.loc[cust_test['Days held']<0,"Days held"]=0
cust_train.loc[cust_train['Days held']<0,"Days held"]=0


# In[1300]:


cust_train['Consumer disputed?'].unique()


# In[1301]:


np.min(cust_test['Days held']),np.min(cust_train['Days held'])

Drop Unnecessary Columns for the Model Building like:'Company', 'State', 'Year_Received', 'Days_held'
# In[1302]:


cust_test=cust_test.drop(['Company','State','Year','Days held'],axis=1)
cust_train=cust_train.drop(['Company','State','Year','Days held'],axis=1)


# In[1303]:


cust_test.head()


# In[1304]:


cust_train.tail()

Change Consumer Disputed Column to 0 and 1(yes to 1, and no to 0)
# In[1306]:


cust_train['Consumer disputed?']=cust_train['Consumer disputed?'].map({'Yes':1,'No':0})


# In[1313]:


cust_train.shape

Create Dummy Variables for categorical features and concat with the original data frame like: 'Product,’ 'Submitted via,’ 'Company response to consumer,’ 'Timely response?'
# In[1308]:


test_dum=pd.get_dummies(cust_test,columns=['Product','Submitted via','Company response to consumer', 'Timely response?'])
train_dum=pd.get_dummies(cust_train,columns=['Product','Submitted via','Company response to consumer', 'Timely response?'])


# In[1316]:


train_dum.head(2)


# In[1318]:


test_dum.head(2)

Scaling the Data Sets (note: discard dependent variable before doing standardization) 
and Make feature Selection with the help of PCA up to 80% of the information.
# In[1319]:


x_train=train_dum.drop(['Consumer disputed?','Issue'],axis=1)
y_train=train_dum['Consumer disputed?']


# In[1321]:


sc=StandardScaler()
x_train_scaled=sc.fit_transform(x_train)
x_train_scaled


# In[1323]:


x_test=test_dum.drop(['Issue'],axis=1)
x_test_scaled=sc.fit_transform(x_test)


# In[1324]:


x_test_scaled


# In[1325]:


pca=PCA(n_components=0.8)
principal_component_train=pca.fit_transform(x_train_scaled)
principal_component_test=pca.fit_transform(x_test_scaled)


# In[1329]:


name=['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11','pc12','pc13','pc14','pc15','pc16','pc17','pc18']
pdf_train=pd.DataFrame(data=principal_component_train,columns=name)
pdf_train.head(2)


# In[1330]:


pdf_test=pd.DataFrame(data=principal_component_test,columns=name)
pdf_test.head(2)

Splitting the Data Sets Into X and Y by the dependent and independent variables (data selected by PCA)
# In[1358]:


x_test=pdf_test


# In[1348]:


x=pdf_train
y=train_dum['Consumer disputed?']
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.3,random_state=10)

Build given models and measure their test and validation accuracy:
o LogisticRegression
o DecisionTreeClassifier
o RandomForestClassifier
o AdaBoostClassifier
o GradientBoostingClassifier
o KNeighborsClassifier
o XGBClassifier
# In[1349]:


#Logistic Regression
lr=LogisticRegression()
lr.fit(x_train,y_train)
yplr=lr.predict(x_val)
print("Logistic Regression:",accuracy_score(y_val,yplr))


# In[1350]:


#DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=1)
dt.fit(x_train,y_train)
ypdt=dt.predict(x_val)
print("DecisionTree Classifier:",accuracy_score(y_val,yplr))


# In[1351]:


#RandomForestClassifier
rd=RandomForestClassifier()
rd.fit(x_train,y_train)
yprd=rd.predict(x_val)
print("Random Forest Classifier:",accuracy_score(y_val,yprd))


# In[1352]:


#AdaBoostClassifier
addt=AdaBoostClassifier(n_estimators=100,learning_rate=0.6)
addt.fit(x_train,y_train)
yaddt=addt.predict(x_val)
print("AdaBoost Classifier:",accuracy_score(y_val,yaddt))


# In[1354]:


#GradientBoostingClassifier
gbc=GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
gbc.fit(x_train,y_train)
ygbc=gbc.predict(x_val)
print("GradientBoosting:",accuracy_score(y_val,ygbc))


# In[1355]:


#KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
yknn=knn.predict(x_val)
print("KNN",accuracy_score(y_val,yknn))


# In[1356]:


#XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train,y_train)
yxgb=xgb.predict(x_val)
print("XGBClassifier",accuracy_score(y_val,yxgb))

Whoever gives the most accurate result uses it and predicts the outcome for the test file 
and fills its dispute column so the business team can take some action accordingly.
XGBClassifier is the best.
# In[1357]:


prediction=xgb.predict(pdf_test)

Export prediction to CSV
# In[1359]:


x_test['Consumer disputed']=prediction


# In[1360]:


x_test.to_csv('new_created.csv')

