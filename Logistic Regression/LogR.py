# Logistic Regression on Weather data to predict if it rains the next day or not
# %%
#Imports needed
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv("weatherAUS.csv")
df
# %%
df.info()
# %%
# As The target column is RainTomorrow we would have to drop the columns where Raintoday and RainTomorrow has missing values

df.dropna(subset = ['RainToday', 'RainTomorrow'], inplace = True)
# %%

# Exploratory Data Analysis and Visualization

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10,6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# %%
px.histogram(df,
             x='Location',
             title = 'Location vs. Rainy Days',
             color = 'RainToday')

#%%
px.histogram(df,
             x='Temp3pm',
             title = 'Temp at 3pm vs. Rain tomorrow',
             color = 'RainTomorrow')
# %%
px.histogram(df,
             x='RainTomorrow',
             color='RainToday',
             title = 'Rain Today vs Rain Tomorrow')

# This is called Class Imbalance, You have two classes 'No' and 'Yes' and you do not have equal no. of
# Observations in each class 
# %%
px.scatter(df.sample(2000),
           title = 'Min Temp vs Max Temp',
           x = 'MinTemp',
           y = 'MaxTemp',
           color = 'RainToday')
# %%
px.strip(df.sample(2000),
         title = 'Temp at 3pm vs Humidity at 3pm',
         x = 'Temp3pm',
         y = 'Humidity3pm',
         color = 'RainTomorrow')
# %%
# Optional -> Working with a Sample
# When working with massive datasets, it's better to work with a sample initially
# Set the value of use_Sample to True when u want to use sample

# use_sample = False
# sample_fraction = 0.1
# if use_sample:
#     df = df.sample(frac = sample_fraction).copy()
# %%

# Training, Validation and Test Sets

# While building real-world machine learning models ,it is quite common to split the dataset into three parts

# Training set - used to train the model, i.e., compute the loss and adjust the model's weights using
# an optimization technique.
 
# Validation set - used to evaluate the model during training, tune model hyperparameters
# (optimization techniques, regularization etc.), and pick the best version of the model. Picking a 
# good validation set is essential for training models that generalize well

# Test set - used to compare different models or approaches and report the model's final accuracy.
# For many datasets, test sets are provided seperately. The test set should reflect the kind of data
# the model will encounter in the real world, as closely as feasible.

# %%

train_val_df, test_df = train_test_split(df, test_size = 0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

# Benefit of random state is that it gives you the guarantee that every time you pass 42 as random_state
# the same 20% fraction will be picked as the test set

# %%
print('train_df.shape: ', train_df.shape)
print('val_df.shape: ', val_df.shape)
print('test_df.shape: ', test_df.shape)

# %%
# However, while working with dates, it's often a better idea to seperate the training,validation and 
# test sets with time, so that the model is trained on data from  the past and evaluated on data from
# the future.
plt.title('No. of Rows per year')
sns.countplot(x=pd.to_datetime(df.Date).dt.year); 

# %%
year = pd.to_datetime(df.Date).dt.year

train_df = df[year < 2015]
val_df = df[year == 2015]
test_df = df[year > 2015]

# %%
# Identifying Input and Target columns
# Not all columns are useful for traing a model like Date here is not needed
input_cols = list(train_df.columns)[1:-1] # First and Last column is skipped 
target_col = 'RainTomorrow'
print(input_cols)
target_col

# %%
# We can now create inputs and targets for the training, validation and test sets for further processing
# and model training
train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

# %%
# Now identifying which columns are numerical and which ones are categorical. As we'll need to convert
# the categorical to numerical to train the logistic regression model

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()
# %%
train_inputs[numeric_cols].describe()
# %%
train_inputs[categorical_cols].nunique()

# %%
# Inputing Missing Numeric Data
# Machine learning models can't work with missing numerical data. The process of filling missing values
# is called imputation.
# There are several ways to do this, but we'll use the most basic one: replacing missing values with the
# average value in the column using the simpleImputer class from sklearn.impute

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'mean')

# Before imputation, let's check the no. of missing values in each numeric column.
train_inputs[numeric_cols].isna().sum()
# %%
# The first step in imputation is to fit the imputer to the data i.e. compute the chosen statistic(mean) 
# for each column in the dataset
imputer.fit(df[numeric_cols])

# after calling fit, the computed statistic for each column is stored in the statistics_ property of 
# imputer
list(imputer.statistics_)

# %%
# The missing values in the training,validation and test set can be filled using the transform method 
# of imputer

train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

# The missing values are now filled with the mean of each column

# Exercise : Apply some other imputation techniques and observe how they change the results of the model
# https://scikit-learn.org/stable/modules/impute.html
# %%
# Next Step is Scaling Numeric Features
# Scaling numeric features ensures that no particular feature has a disproportionate impact on the model's
# loss. Optimization algorithms also work better in practice with smaller numbers.
# We scale them to a same range of (0,1) or (-1,1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# First we fit the scaler to the data
scaler.fit(df[numeric_cols])

print('Minimum: ')
list(scaler.data_min_)

# %%
print('Maximum: ')
list(scaler.data_max_)

# %%
# We can now seperately scale the data sets using transform method of scaler
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# %%
train_inputs[numeric_cols].describe()

# %%
# Learn more about scaling techniques here: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/

# %%
# Ecvoding Categorical Columns using One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(df[categorical_cols])

# %%
encoder.categories_

# %%
# We can generate column names for each individual category using get_feature_names
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
print(encoded_cols)

# %%
# All of the above columns will be added to the data sets
# OneHotEncoder.transform returns a sparse matrix (depending on settings).
# Convert to dense array so the shape matches the list of encoded column names
train_enc = encoder.transform(train_inputs[categorical_cols])
val_enc = encoder.transform(val_inputs[categorical_cols])
test_enc = encoder.transform(test_inputs[categorical_cols])

# If the encoder returned a sparse matrix, convert to array; otherwise keep as is
try:
    train_arr = train_enc.toarray()
except AttributeError:
    train_arr = train_enc

try:
    val_arr = val_enc.toarray()
except AttributeError:
    val_arr = val_enc

try:
    test_arr = test_enc.toarray()
except AttributeError:
    test_arr = test_enc

train_inputs[encoded_cols] = train_arr
val_inputs[encoded_cols] = val_arr
test_inputs[encoded_cols] = test_arr
# %%
