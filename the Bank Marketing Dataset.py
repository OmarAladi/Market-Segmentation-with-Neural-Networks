#!/usr/bin/env python
# coding: utf-8

# In[1]:


import  warnings
warnings.filterwarnings('ignore')


# # 1: Import Libraries and Load Data

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras import layers
from tensorflow.keras import Sequential


# In[3]:


df = pd.read_csv("bank-additional-full.csv", sep=';')
df.head(5)
# label  -->  y


# In[4]:


df.shape


# # 2: Exploratory Data Analysis (EDA)

# ## Features:
# #### 1) Age (numeric)
# #### 2) Job : type of job (categorical: ‘admin.’, ‘blue-collar’, ‘entrepreneur’, ‘housemaid’, ‘management’, ‘retired’, ‘self-employed’, ‘services’, ‘student’, ‘technician’, ‘unemployed’, ‘unknown’)
# #### 3) Marital : marital status (categorical: ‘divorced’, ‘married’, ‘single’, ‘unknown’ ; note: ‘divorced’ means divorced or widowed)
# #### 4) Education (categorical: ‘basic.4y’, ‘basic.6y’, ‘basic.9y’, ‘high.school’, ‘illiterate’, ‘professional.course’, ‘university.degree’, ‘unknown’)
# #### 5) Default: has credit in default? (categorical: ‘no’, ‘yes’, ‘unknown’)
# #### 6) Housing: has housing loan? (categorical: ‘no’, ‘yes’, ‘unknown’)
# #### 7) Loan: has personal loan? (categorical: ‘no’, ‘yes’, ‘unknown’)
# #### 8) Contact: contact communication type (categorical: ‘cellular’,‘telephone’)
# #### 9) Month: last contact month of year (categorical: ‘jan’, ‘feb’, ‘mar’, …, ‘nov’, ‘dec’)
# #### 10) Day_of_week: last contact day of the week (categorical: ‘mon’,‘tue’,‘wed’,‘thu’,‘fri’)
# #### 11) Duration: last contact duration, in seconds (numeric).
# #### 12) Campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# #### 13) Pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# #### 14) Previous: number of contacts performed before this campaign and for this client (numeric)
# #### 15) Poutcome: outcome of the previous marketing campaign (categorical: ‘failure’,‘nonexistent’,‘success’)
# #### 16) Emp.var.rate: employment variation rate - quarterly indicator (numeric)
# #### 17) Cons.price.idx: consumer price index - monthly indicator (numeric)
# #### 18) Cons.conf.idx: consumer confidence index - monthly indicator (numeric)
# #### 19) Euribor3m: euribor 3 month rate - daily indicator (numeric)
# #### 20) Nr.employed: number of employees - quarterly indicator (numeric) Output variable (desired target):
# #### 21) y: has the client subscribed a term deposit? (binary: ‘yes’, ‘no’)

# In[5]:


df.info()


# In[6]:


#Checking out the categories and their respective counts in each feature
print("Job:",df.job.value_counts(),sep = '\n')
print("-"*40)
print("Marital:",df.marital.value_counts(),sep = '\n')
print("-"*40)
print("Education:",df.education.value_counts(),sep = '\n')
print("-"*40)
print("Default:",df.default.value_counts(),sep = '\n')
print("-"*40)
print("Housing loan:",df.housing.value_counts(),sep = '\n')
print("-"*40)
print("Personal loan:",df.loan.value_counts(),sep = '\n')
print("-"*40)
print("Contact:",df.contact.value_counts(),sep = '\n')
print("-"*40)
print("Month:",df.month.value_counts(),sep = '\n')
print("-"*40)
print("Day:",df.day_of_week.value_counts(),sep = '\n')
print("-"*40)
print("Previous outcome:",df.poutcome.value_counts(),sep = '\n')
print("-"*40)
print("Outcome of this campaign:",df.y.value_counts(),sep = '\n')
print("-"*40)


# In[7]:


df.describe()


# #### Dealing with Missing Values

# In[8]:


df.replace('unknown', np.NaN, inplace=True)


# In[9]:


df.isnull().sum()


# In[10]:


df.dropna(inplace=True)


# #### study 'default'

# In[11]:


df.default.value_counts()


# In[12]:


df.drop("default",axis=1,inplace=True)


# #### study 'pdays'
# There is also a issue in pdays feature. if the value is 999, then it will be replaced with a 0 which means that the client was not previously contacted.

# In[13]:


df.pdays.value_counts()


# In[14]:


df.loc[df['pdays'] == 999, 'pdays'] = 0


# In[15]:


sns.pairplot(df)


# In[16]:


# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns

# Plot histograms for categorical columns
for col in categorical_cols:
    plt.figure(figsize=(6, 3))
    df[col].value_counts().plot(kind='bar')
    plt.title(col)
    plt.show()

# Plot histograms + box plots for numerical columns
for col in numerical_cols:
    plt.figure(figsize=(6, 3))
    
    # Histogram
    plt.subplot(1, 2, 1)
    df[col].hist(bins=30)
    plt.title(col)
    
    # Box plot
    plt.subplot(1, 2, 2)
    df[col].plot(kind='box')
    
    plt.show()


# In[17]:


categorical_cols = df.select_dtypes(include=['object', 'bool']).columns

for col in categorical_cols:
    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x=col, hue='y', palette='GnBu', order=df[col].value_counts().index)
    plt.title(col)
    plt.show()


# # 3: Data Preprocessing

# In[18]:


# Apply binary encoding for the 'y' column (output variable)
df['y'] = df['y'].map({'no': 0, 'yes': 1})

"""
# Convert target variable into numeric
df.y = df.y.map({'no':0, 'yes':1}).astype('uint8')
"""


# In[19]:


le = LabelEncoder()
objects = ["job","marital","education","housing","loan","contact","month","day_of_week", "poutcome"]
for i in objects:  
    df[i] = le.fit_transform(df[i])


# In[20]:


df.head()


# # 4: Split Data into Training and Testing Sets

# In[21]:


# Generate Train and Test Split
x = df.drop(['y'], axis=1)
y = df["y"]
#..........................................
columns = x.columns

scalar = StandardScaler()
x = scalar.fit_transform(x) 

x = pd.DataFrame(x, columns=columns)
#..........................................
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)  # all are df
print("x_train shape = ", x_train.shape)
print("y_train shape = ", y_train.shape)
print("x_test shape = ", x_test.shape)
print("y_test shape = ", y_test.shape)


# # Model 1

# # 5 : Build and Train the Neural Network Model

# In[22]:


model = Sequential()
model.add(Dense(64, activation='relu', input_dim=x_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))


# # 6 : Evaluate the Model

# In[23]:


preds = model.predict(x_test).reshape((-1,))

res = pd.DataFrame()
res["Actual"] = y_test.values
res["Predictian"] = preds
res


# In[24]:


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


# In[25]:


# Generate predictions using the trained model
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)  

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'], 
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# # Model 2

# # 5 : Build and Train the Neural Network Model

# In[30]:


model2 = Sequential()

model2.add(layers.Dense(32, activation="sigmoid", input_shape=(x_train.shape[1],)))
model2.add(layers.Dense(16, activation="sigmoid"))
model2.add(layers.Dense(1, activation="sigmoid"))
model2.summary()


# In[31]:


# Compile the model
model2.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
# Train the model
model2.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)


# # 6 : Evaluate the Model

# In[40]:


preds = model2.predict(x_test).reshape((-1,))

res = pd.DataFrame()
res["Actual"] = y_test.values
res["Predictian"] = preds
res


# In[39]:


# Generate predictions using the trained model
y_pred = model2.predict(x_test)
y_pred = (y_pred > 0.25)  

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'], 
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

