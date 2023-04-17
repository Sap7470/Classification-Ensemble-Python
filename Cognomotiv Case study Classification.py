#!/usr/bin/env python
# coding: utf-8

# In[179]:


#Importing packages#
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import statsmodels.api as sm



from sklearn.linear_model import LogisticRegression
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report


# In[180]:


#Fig size etc #
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
import warnings
warnings.filterwarnings('ignore')


# In[181]:


# Data Loading and Exploration

df= pd.read_csv("/Users/saptarshichakraberty/Downloads/simple_classification_or_regression.csv")

print(df.shape) 

print(df.columns)
df.head(5)


# In[182]:


print(df.info()) 
print(df.describe())


# In[183]:


# Min of income is -654, so we will drop that data point, as we have enough datapoints in the dataset
# locate the rows with negative values in the 'Income' column
neg_rows = df[df['Income'] < 0]
# print the negative rows
print(neg_rows) #245

# drop the row with index as neg_rows
df = df.drop(index= 245)
df = df.reset_index(drop=True)


# In[184]:


#print(df[df['Number'] ==246]) # Not showing any, which means it is deleted
print(df.describe())
df= df.drop('Number', axis= 1)
print(df.columns)


# In[185]:


# Checking missing observation
print("Number of null values in the data set are - ",df.isnull().values.any().sum())


# In[186]:


print(df['City'].value_counts()) 
print('-'*20)
print(df['Gender'].value_counts()) # M: 83800; F:66199
print('-'*20)
print(df['Illness'].value_counts()) #No(0): 137861; Yes(1):12138


# In[187]:


# create the box plot for the 'Age' column
fig, ax = plt.subplots()
df.boxplot(column='Age', ax=ax)

# set the title and y-axis label
ax.set_title('Box Plot of Age Column')
ax.set_ylabel('Age')

## Plot doesnt show any outlier


# In[188]:


# create the box plot for the 'Income' column
fig, ax = plt.subplots()
df.boxplot(column='Income', ax=ax)

# set the title and y-axis label
ax.set_title('Box Plot of Income Column')
ax.set_ylabel('Income')
# There are outliers


# In[189]:


# identify extreme values in the 'Income' column
q_low = df['Income'].quantile(0.25)
q_high = df['Income'].quantile(0.75)
iqr= q_high- q_low # inter quartile range
lob= q_low - 1.5*iqr # lower outlier boundary
uob= q_high + 1.5*iqr # upper outlier boundary

# remove rows with extreme values in the 'Income' column
df1 = df[(df['Income'] >= lob) & (df['Income'] <= uob)]
print(df.shape) #(149999, 5)
print(df1.shape) # (135217, 5)


'''# remove rows with extreme values in the 'Income' column
df2 = df[(df['Income'] >= df['Income'].quantile(0.15)) & (df['Income'] <= df['Income'].quantile(0.90))]
print(df2.shape)'''# We did this later


# In[190]:


# create the box plot for the 'Income' column
fig, ax = plt.subplots()
df1.boxplot(column='Income', ax=ax)

# set the title and y-axis label
ax.set_title('Box Plot of Income Column after trimming')
ax.set_ylabel('Income')
# There are outliers


# In[191]:


print(df1['Gender'].value_counts()) # M: 83800; F:66199 is now, M: 76276; F: 58941
print(df1['Illness'].value_counts()) #No(0): 137861; Yes(1):12138 is now, 0: 124261, 1:10956


# In[192]:


df1 = df1.drop('City', axis=1) # Removing "City", we are not going to use the location for the initial modeling scheme
print(df1.head())


# In[193]:


# Encoding Object variables 
#CONVERT: convert objects to category using Label Encoder for train and test/validation dataset

#code categorical data
label = LabelEncoder()
df1['Gender']= label.fit_transform(df1['Gender'])
df1['Illness']= label.fit_transform(df1['Illness'])
df1.head()


# In[194]:


sns.distplot(df1.Age,bins=20) #More or less all age categories has same number of observation
plt.show()


# In[195]:


sns.distplot(df1.Income,bins=20) 
plt.show()


# In[196]:


sns.heatmap(df1.corr(),
           vmax= 1, vmin= -1, center= 0,
           cmap= sns.diverging_palette(20,220, n= 200),
           annot= True)
# Note: None of the independent variables are correlated


# In[197]:


df1['Illness'].value_counts()/len(df1['Illness'])


# In[198]:


Target = ['Illness']
X= df1.drop('Illness', axis= 1)
y= df1['Illness']
X_train, X_test, y_train, y_test= model_selection.train_test_split(X, y, test_size= 0.3, random_state= 1)


print(X_train.shape) # (94651, 3)#
print('-'*20)
print(X_test.shape) # (40566, 3) #
X.columns #['Gender', 'Age', 'Income']


# In[199]:


print(y_train.value_counts())#0: 87094, 1:7557
print(y_test.value_counts()) #0: 37167, 1:3399


# In[200]:


# Standardization:
sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)
print(X_train.shape) # (94651, 3) #
print(X_test.shape) # (40566, 3) #


# In[ ]:


## Before balancing the data


# In[201]:


X_train1 = sm.add_constant(X_train)
logit_model=sm.Logit(y_train,X_train1)
result=logit_model.fit()
print("The intercept b0= ", result.params[0])
print(result.summary()) # None of them are significant


# In[202]:


#Checking linearity of the log odds with the independent variables
import numpy as np

# Get the predicted probabilities for X_train
y_pred_train = result.predict(X_train1)

# Find the correlation matrix between X_train and y_pred
correlation_matrix = np.corrcoef(X_train1.T, y_pred_train)

# Print the correlation matrix
print(correlation_matrix)

len(y_pred_train)


# In[203]:


# Plotting errors to see if they are independent
# Calculate residuals
residuals_train = y_train - y_pred_train

# Plot residuals against predicted probabilities
plt.scatter(y_pred_train, residuals_train)
plt.xlabel('Predicted Probability')
plt.ylabel('Residuals')
plt.show()


# In[205]:


X_test1 = sm.add_constant(X_test)


# Make prediction for the test data
y_pred_proba = result.predict(X_test1)
y_pred = (y_pred_proba >= 0.5).astype(int)


from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate the classification report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# Calculate the ROC AUC score
auc = roc_auc_score(y_test, y_pred)
print("ROC AUC Score:", auc)

# We can see that the model is predicting everybody as 0
print(y_pred_proba.mean()) #0.0798
print(y_pred_proba.std()) # 0.00118
print(max(y_pred_proba)) #0.083497
print(min(y_pred_proba)) #0.0762634


# In[206]:


# Lets change the cutoff to see if that helps
y_pred = (y_pred_proba >= 0.0798).astype(int)


from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate the classification report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# Calculate the ROC AUC score
auc = roc_auc_score(y_test, y_pred)
print("ROC AUC Score:", auc)


# In[ ]:


# Didn't help much. Now we will use the other classification techniquesa and at the end we will balance the data synthetically and fit the models again


# In[207]:


# k-nearest neighbors
range_k = range(1,16)
scores = {}
scores_list = []
for k in range_k:
   classifier = KNeighborsClassifier(n_neighbors=k)
   classifier.fit(X_train, y_train)
   y_pred = classifier.predict(X_test)
   scores[k] = metrics.accuracy_score(y_test,y_pred)
   scores_list.append(metrics.accuracy_score(y_test,y_pred))
    
# Find the item with the maximum value
max_item = max(scores, key=scores.get)
print(max_item)
print(scores)


# In[208]:


classifier = KNeighborsClassifier(n_neighbors=max_item)
classifier.fit(X_train, y_train)
# Make prediction for the test data
y_pred= classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate the classification report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# Calculate the ROC AUC score
auc = roc_auc_score(y_test, y_pred)
print("ROC AUC Score:", auc)


# In[209]:


ML_algos = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier()
    ]

for alg in ML_algos:
    model = alg  
    model.fit(X_train, y_train )  
    y_pred = model.predict(X_test)
    print(model)
    print("-"*50)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Calculate the classification report
    cr = classification_report(y_test, y_pred)
    print("Classification Report:\n", cr)

    # Calculate the ROC AUC score
    auc = roc_auc_score(y_test, y_pred)
    print("ROC AUC Score:", auc)


# In[210]:


#xgBoost is sensetive to imbalance class, still will try before balancing
model = XGBClassifier()  
model.fit(X_train, y_train)
print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# evaluate predictions
print('XgBoost')
print("-"*20)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate the classification report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# Calculate the ROC AUC score
auc = roc_auc_score(y_test, y_pred)
print("ROC AUC Score:", auc)


# In[ ]:





# In[ ]:





# In[211]:


# Undersampling: randomly removing examples from the majority class to balance the class distribution.
from imblearn.under_sampling import RandomUnderSampler

# Define the undersampling object
under_sampler = RandomUnderSampler(sampling_strategy='majority')

# Apply the undersampling on X_train and y_train
X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_train, y_train)


# In[212]:


X_train1 = sm.add_constant(X_train_resampled)
logit_model=sm.Logit(y_train_resampled,X_train1)
result=logit_model.fit()
print("The intercept b0= ", result.params[0])
print(result.summary()) # Still none of them are significant


# In[213]:


# Get the predicted probabilities for X_train
y_pred_train = result.predict(X_train1)

# Find the correlation matrix between X_train and y_pred
correlation_matrix = np.corrcoef(X_train1.T, y_pred_train)

# Print the correlation matrix
print(correlation_matrix)

len(y_pred_train)


# In[214]:


# Calculate residuals
residuals_train = y_train_resampled - y_pred_train

# Plot residuals against predicted probabilities
plt.scatter(y_pred_train, residuals_train)
plt.xlabel('Predicted Probability')
plt.ylabel('Residuals')
plt.show()


# In[215]:


X_test1 = sm.add_constant(X_test)


# Make prediction for the test data
y_pred_proba = result.predict(X_test1)
y_pred = (y_pred_proba >= 0.5).astype(int)


from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate the classification report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# Calculate the ROC AUC score
auc = roc_auc_score(y_test, y_pred)
print("ROC AUC Score:", auc)


y_pred_proba.mean() 


# In[216]:


# k-nearest neighbors: X_train_resampled, y_train_resampled
range_k = range(1,16)
scores = {}
scores_list = []
for k in range_k:
   classifier = KNeighborsClassifier(n_neighbors=k)
   classifier.fit(X_train_resampled, y_train_resampled)
   y_pred = classifier.predict(X_test)
   scores[k] = metrics.accuracy_score(y_test,y_pred)
   scores_list.append(metrics.accuracy_score(y_test,y_pred))
    
# Find the item with the maximum value
max_item = max(scores, key=scores.get)
print(max_item)
print(scores)
#########

classifier = KNeighborsClassifier(n_neighbors=max_item)
classifier.fit(X_train, y_train)
# Make prediction for the test data
y_pred= classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate the classification report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# Calculate the ROC AUC score
auc = roc_auc_score(y_test, y_pred)
print("ROC AUC Score:", auc)


# In[217]:


#X_train_resampled, y_train_resampled

ML_algos = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier()
    ]

for alg in ML_algos:
    model = alg  
    model.fit(X_train_resampled, y_train_resampled)  
    y_pred = model.predict(X_test)
    print(model)
    print("-"*50)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Calculate the classification report
    cr = classification_report(y_test, y_pred)
    print("Classification Report:\n", cr)

    # Calculate the ROC AUC score
    auc = roc_auc_score(y_test, y_pred)
    print("ROC AUC Score:", auc)


# In[218]:


#X_train_resampled, y_train_resampled

model = XGBClassifier()  
model.fit(X_train_resampled, y_train_resampled)
print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# evaluate predictions
print('XgBoost')
print("-"*20)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate the classification report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# Calculate the ROC AUC score
auc = roc_auc_score(y_test, y_pred)
print("ROC AUC Score:", auc)


# In[ ]:





# In[219]:


# Oversampling: creating synthetic examples in the minority class to balance the class distribution.
#from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

# Define the oversampling object
#over_sampler = RandomOverSampler(sampling_strategy='minority')
# or 
over_sampler = SMOTE()

# Apply the oversampling on X_train and y_train
X_train_resampled, y_train_resampled = over_sampler.fit_resample(X_train, y_train)


# In[220]:


X_train1 = sm.add_constant(X_train_resampled)
logit_model=sm.Logit(y_train_resampled,X_train1)
result=logit_model.fit()
print("The intercept b0= ", result.params[0])
print(result.summary()) # Income is significant at 0.05 level


# In[221]:


# Get the predicted probabilities for X_train
y_pred_train = result.predict(X_train1)

# Find the correlation matrix between X_train and y_pred
correlation_matrix = np.corrcoef(X_train1.T, y_pred_train)

# Print the correlation matrix
print(correlation_matrix) # Income is highly correlated to p_hats

len(y_pred_train)


# In[222]:


# Calculate residuals
residuals_train = y_train_resampled - y_pred_train

# Plot residuals against predicted probabilities
plt.scatter(y_pred_train, residuals_train)
plt.xlabel('Predicted Probability')
plt.ylabel('Residuals')
plt.show()


# In[223]:


X_test1 = sm.add_constant(X_test)


# Make prediction for the test data
y_pred_proba = result.predict(X_test1)
y_pred = (y_pred_proba >= 0.5).astype(int)


from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate the classification report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# Calculate the ROC AUC score
auc = roc_auc_score(y_test, y_pred)
print("ROC AUC Score:", auc)


y_pred_proba.mean() #0.49997
y_pred_proba.std() # 0.003256
max(y_pred_proba) #0.5092
min(y_pred_proba) #0.4903400


# In[224]:


# k-nearest neighbors: X_train_resampled, y_train_resampled
range_k = range(1,16)
scores = {}
scores_list = []
for k in range_k:
   classifier = KNeighborsClassifier(n_neighbors=k)
   classifier.fit(X_train_resampled, y_train_resampled)
   y_pred = classifier.predict(X_test)
   scores[k] = metrics.accuracy_score(y_test,y_pred)
   scores_list.append(metrics.accuracy_score(y_test,y_pred))
    
# Find the item with the maximum value
max_item = max(scores, key=scores.get)
print(max_item)
print(scores)
#########

classifier = KNeighborsClassifier(n_neighbors=max_item)
classifier.fit(X_train, y_train)
# Make prediction for the test data
y_pred= classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate the classification report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# Calculate the ROC AUC score
auc = roc_auc_score(y_test, y_pred)
print("ROC AUC Score:", auc)


# In[225]:


ML_algos = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier()
    ]

for alg in ML_algos:
    model = alg  
    model.fit(X_train_resampled, y_train_resampled)  
    y_pred = model.predict(X_test)
    print(model)
    print("-"*50)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Calculate the classification report
    cr = classification_report(y_test, y_pred)
    print("Classification Report:\n", cr)

    # Calculate the ROC AUC score
    auc = roc_auc_score(y_test, y_pred)
    print("ROC AUC Score:", auc)


# In[226]:


model = XGBClassifier()  
model.fit(X_train_resampled, y_train_resampled)
print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# evaluate predictions
print('XgBoost')
print("-"*20)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate the classification report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# Calculate the ROC AUC score
auc = roc_auc_score(y_test, y_pred)
print("ROC AUC Score:", auc)


# In[ ]:





# In[227]:


#Even after oversampling and undersampling the model performances are not adequet
#We noticed there are still some outliers left. So remove rows with extreme values in the 'Income' column
df2 = df[(df['Income'] >= df['Income'].quantile(0.15)) & (df['Income'] <= df['Income'].quantile(0.90))]
print(df2.shape) #(112501, 5)
print(df.shape) #(149999, 5)
print(df1.shape) # (135217, 5)


# In[228]:


# create the box plot for the 'Income' column
fig, ax = plt.subplots()
df2.boxplot(column='Income', ax=ax)

# set the title and y-axis label
ax.set_title('Box Plot of Income Column after second trimming')
ax.set_ylabel('Income')
# There are no outliers


# In[229]:


print(df2['Gender'].value_counts()) # M: 83800; F:66199 to M: 76276; F: 58941 to M:62842, F:49659
print(df2['Illness'].value_counts()) #No(0): 137861; Yes(1):12138 to 0: 124261, 1:10956 to 0:103448, 1:9053

df2 = df2.drop('City', axis=1) # Removing "City", we are not going to use the location for the initial modeling scheme
print(df2.head())


# In[230]:


# Encoding Object variables 
#CONVERT: convert objects to category using Label Encoder for train and test/validation dataset

#code categorical data
label = LabelEncoder()
df2['Gender']= label.fit_transform(df2['Gender'])
df2['Illness']= label.fit_transform(df2['Illness'])
df2.head()


# In[231]:


sns.distplot(df2.Age,bins=20) 
plt.show()


# In[232]:


sns.distplot(df2.Income,bins=20) 
plt.show()


# In[233]:


sns.heatmap(df2.corr(),
           vmax= 1, vmin= -1, center= 0,
           cmap= sns.diverging_palette(20,220, n= 200),
           annot= True)
# Note: None of the independent variables are correlated


# In[234]:


df2['Illness'].value_counts()/len(df2['Illness'])


# In[235]:


Target = ['Illness']
X= df2.drop('Illness', axis= 1)
y= df2['Illness']
X_train, X_test, y_train, y_test= model_selection.train_test_split(X, y, test_size= 0.3, random_state= 1)


print(X_train.shape) # (94651, 3) to (78750, 3)#
print('-'*20)
print(X_test.shape) # (40566, 3) to (33751, 3) #
X.columns #['Gender', 'Age', 'Income']


# In[236]:


print(y_train.value_counts())#0: 87094, 1:7557 to 0:72393, 1: 6357
print(y_test.value_counts()) #0: 37167, 1:3399 to 0:31055, 1:2696


# In[237]:


# Standardization:
sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)
print(X_train.shape) # (78750, 3) #
print(X_test.shape) # (33751, 3) #


# In[238]:


pwd


# In[ ]:


# Balancing the data using oversampling


# In[239]:


# Oversampling: creating synthetic examples in the minority class to balance the class distribution.
from imblearn.over_sampling import SMOTE

# Define the oversampling object
over_sampler = SMOTE()

# Apply the oversampling on X_train and y_train
X_train_resampled, y_train_resampled = over_sampler.fit_resample(X_train, y_train)

X_train_resampled.shape #(144786, 3) ==== 72393*2= 144786


# In[240]:


X_train1 = sm.add_constant(X_train_resampled)
logit_model=sm.Logit(y_train_resampled,X_train1)
result=logit_model.fit()
print("The intercept b0= ", result.params[0])
print(result.summary()) # Still none of them are significant


# In[241]:


# Get the predicted probabilities for X_train
y_pred_train = result.predict(X_train1)

# Find the correlation matrix between X_train and y_pred
correlation_matrix = np.corrcoef(X_train1.T, y_pred_train)

# Print the correlation matrix
print(correlation_matrix) # Income is highly correlated to p_hats

len(y_pred_train)


# In[242]:


# Calculate residuals
residuals_train = y_train_resampled - y_pred_train

# Plot residuals against predicted probabilities
plt.scatter(y_pred_train, residuals_train)
plt.xlabel('Predicted Probability')
plt.ylabel('Residuals')
plt.show()


# In[243]:


X_test1 = sm.add_constant(X_test)


# Make prediction for the test data
y_pred_proba = result.predict(X_test1)
y_pred = (y_pred_proba >= 0.5).astype(int)


from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate the classification report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# Calculate the ROC AUC score
auc = roc_auc_score(y_test, y_pred)
print("ROC AUC Score:", auc)

print('-'*30)
print(y_pred_proba.mean()) #0.49997
print(y_pred_proba.std()) # 0.003256
print(max(y_pred_proba)) #0.5092
print(min(y_pred_proba)) #0.4903400


# In[244]:


# k-nearest neighbors: X_train_resampled, y_train_resampled
range_k = range(1,16)
scores = {}
scores_list = []
for k in range_k:
   classifier = KNeighborsClassifier(n_neighbors=k)
   classifier.fit(X_train_resampled, y_train_resampled)
   y_pred = classifier.predict(X_test)
   scores[k] = metrics.accuracy_score(y_test,y_pred)
   scores_list.append(metrics.accuracy_score(y_test,y_pred))
    
# Find the item with the maximum value
max_item = max(scores, key=scores.get)
print(max_item)
print(scores)
#########

classifier = KNeighborsClassifier(n_neighbors=max_item)
classifier.fit(X_train, y_train)
# Make prediction for the test data
y_pred= classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate the classification report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# Calculate the ROC AUC score
auc = roc_auc_score(y_test, y_pred)
print("ROC AUC Score:", auc)


# In[245]:


ML_algos = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier()
    ]

for alg in ML_algos:
    model = alg  
    model.fit(X_train_resampled, y_train_resampled)  
    y_pred = model.predict(X_test)
    print(model)
    print("-"*50)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Calculate the classification report
    cr = classification_report(y_test, y_pred)
    print("Classification Report:\n", cr)

    # Calculate the ROC AUC score
    auc = roc_auc_score(y_test, y_pred)
    print("ROC AUC Score:", auc)


# In[246]:


model = XGBClassifier()  
model.fit(X_train_resampled, y_train_resampled)
print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# evaluate predictions
print('XgBoost')
print("-"*20)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate the classification report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# Calculate the ROC AUC score
auc = roc_auc_score(y_test, y_pred)
print("ROC AUC Score:", auc)


# In[ ]:




