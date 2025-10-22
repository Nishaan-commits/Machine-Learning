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
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(df[categorical_cols])

# %%
encoder.categories_

# %%
# We can generate column names for each individual category using get_feature_names
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
print(encoded_cols)

# %%
# All of the above columns will be added to the data sets
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

# %%
# When dataframe gets highly fragmented run these lines to defragment it
# train_inputs = train_inputs.copy()
# val_inputs = val_inputs.copy()
# test_inputs = test_inputs.copy()    
# %%    
# Let's see if these columns have been added
pd.set_option('display.max_columns', None)
test_inputs.head()

# %%
# Saving Processed Data to Disk
# It can be useful to save the processed data to disk, so that we don't have to repeat the preprocessing
# every time you open the notebook. The parquet format is a fast and efficient format for saving and
# loading Pandas dataframes

print('train_inputs: ', train_inputs.shape)
print('train_targets: ', train_targets.shape)
print('val_inputs: ', val_inputs.shape)
print('val_targets: ', val_targets.shape)
print('test_inputs: ', test_inputs.shape)
print('test_targets: ', test_targets.shape)

# %%
train_inputs.to_parquet('train_inputs.parquet') 
val_inputs.to_parquet('val_inputs.parquet')
test_inputs.to_parquet('test_inputs.parquet')

# %%
pd.DataFrame(train_targets).to_parquet('train_targets.parquet')
pd.DataFrame(val_targets).to_parquet('val_targets.parquet')
pd.DataFrame(test_targets).to_parquet('test_targets.parquet')

# %%

#We can read the data back using pd.read_parquet
train_inputs = pd.read_parquet('train_inputs.parquet')
val_inputs = pd.read_parquet('val_inputs.parquet')
test_inputs = pd.read_parquet('test_inputs.parquet')

train_targets = pd.read_parquet('train_targets.parquet')[target_col]
val_targets = pd.read_parquet('val_targets.parquet')[target_col]
test_targets = pd.read_parquet('test_targets.parquet')[target_col]

# %%
# The benefit of saving the processed data is that we can skip all the preprocessing steps and directly
# load the data from disk whenever we want to work with it
# Basically When you have done a lot of preprocessing just save the outputs after the preprocessing

# %%
# Training a Logistic Regression Model
# Logistic Regression is a simple yet powerful model for binary classification tasks.In a Logistic 
# Regression model:

#1 we take linear combination (or weighted sum) of the input features
#2 we apply the sigmoid function to the linear combination to get a probability value between 0 and 1
#3 this number represents the probability of the input being classified as "Yes" (e.g., 'Yes' for RainTomorrow)
#4 instead of RMSE, the cross entropy loss function is used to evaluate the results

# %%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')
model.fit(train_inputs[numeric_cols + encoded_cols], train_targets)

# %%
# Let's check the weights and biases of the trained model
coef_table = pd.DataFrame({
    'Feature': numeric_cols + encoded_cols,
    'Coefficient': model.coef_[0]
})
print(coef_table.sort_values(by='Coefficient', ascending=False))
print(model.intercept_)
# %%

sns.barplot(data=coef_table.sort_values(by='Coefficient', ascending=False).head(10),y='Feature', x='Coefficient')

# %%
# Making Predictions and Evaluating the Model
# We can use the trained model to make predictions on the validation and test sets using the predict method

X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

train_preds = model.predict(X_train) 

# %%
train_preds
 
# %%
train_targets

# %%
# We can output a probabilistic prediction using the predict_proba method
train_probs = model.predict_proba(X_train)
train_probs
# The numbers above indicate the probabilities for the target classes "No" and 'Yes' respectively

# %%
# We can test the accuracy of the model's predictions by computing the percentage of matching values
# in train_preds and train_targets

from sklearn.metrics import accuracy_score
accuracy_score(train_targets, train_preds)
# %%

from sklearn.metrics import confusion_matrix
confusion_matrix(train_targets, train_preds, normalize='true')

# %%
# Let's define a helper function to generate predictions, compute the accuracy and plot a confusion
# matrix for a given set of inputs

def predict_and_plot(inputs,targets, name=''):
    preds = model.predict(inputs)
    acc = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(acc * 100))
    
    cm = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cm, annot=True) 
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title(f'Confusion Matrix for {name} set')
    
    return preds
# %%
train_preds = predict_and_plot(X_train, train_targets, name='Training')
# %%
# Let's compute the model's accuracy on validation and test sets
val_preds = predict_and_plot(X_val, val_targets, name='Validation')
# %%
test_preds = predict_and_plot(X_test, test_targets, name='Test')
# %%
# The accuracy of the model on the test and validation set are above 84%, which suggests that our model
# generalizes well to unseen data.
# But how good is 84% accuracy really? While this depends on the nature of problem and on business
# requirements, a good way to verify whether a model has actually learned something useful is to 
# compare its results to a "random" or "dumb" model.
# Creating 2 dumb models for comparison

def random_guess(inputs):
    return np.random.choice(['No','Yes'], size=len(inputs))

def all_no(inputs):
    return np.full(len(inputs), 'No')
# %%
accuracy_score(test_targets, random_guess(X_test))
# %%
accuracy_score(test_targets, all_no(X_test))
# %%
# Our model performs significantly better than both dumb models, indicating that it has learned useful
# patterns from the data.


# %%
# Making Predictions on a single input
# Once the model has been trained to a satisfactory accuracy, it can be used to make predictions on new data.
# Consider the following dictionary containing data collected from the Katherine weather department today.

new_input = {'Date' : '2021-06-19',
             'Location' : 'Katherine',
                'MinTemp' : 23.2,
                'MaxTemp' : 33.2,
                'Rainfall' : 10.2,
                'Evaporation' : 4.2,
                'Sunshine' : np.nan,
                'WindGustDir' : 'NNW',
                'WindGustSpeed' : 52.0,
                'WindDir9am' : 'NW',
                'WindDir3pm' : 'NNE',
                'WindSpeed9am' : 13.0,
                'WindSpeed3pm' : 20.0,
                'Humidity9am' : 89.0,
                'Humidity3pm' : 58.0,
                'Pressure9am' : 1004.8,
                'Pressure3pm' : 1001.5,
                'Cloud9am' : 8.0,
                'Cloud3pm' : 5.0,
                'Temp9am' : 25.7,
                'Temp3pm' : 33.0,
                'RainToday' : 'Yes'}

# The first step is to convert this dictionary into a dataframe.
new_input_df = pd.DataFrame([new_input])
new_input_df

# %%
# We must apply the same transformations applied while training the model:
#1 Imputation of missing values
#2 scaling numeric features
#3 encoding categorical features

new_input_df[numeric_cols] = imputer.transform(new_input_df[numeric_cols])
new_input_df[numeric_cols] = scaler.transform(new_input_df[numeric_cols])
new_input_df[encoded_cols] = encoder.transform(new_input_df[categorical_cols])
# %%
X_new_input = new_input_df[numeric_cols+encoded_cols]
X_new_input
# %%
# We can now make a prediction
prediction =model.predict(X_new_input)[0]
prediction

# %%
prob = model.predict_proba(X_new_input)[0]
prob
# %%
# Helper Function to make predictions on new inputs
def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    
    return pred, prob   
# %%
single_input = {'Date' : '2021-06-19',
             'Location' : 'Perth',
                'MinTemp' : 23.2,
                'MaxTemp' : 33.2,
                'Rainfall' : 10.2,
                'Evaporation' : 4.2,
                'Sunshine' : np.nan,
                'WindGustDir' : 'NNW',
                'WindGustSpeed' : 50.0,
                'WindDir9am' : 'NW',
                'WindDir3pm' : 'NNE',
                'WindSpeed9am' : 13.0,
                'WindSpeed3pm' : 20.0,
                'Humidity9am' : 89.0,
                'Humidity3pm' : 58.0,
                'Pressure9am' : 1000.8,
                'Pressure3pm' : 1001.5,
                'Cloud9am' : 8.0,
                'Cloud3pm' : 5.0,
                'Temp9am' : 20.7,
                'Temp3pm' : 33.0,
                'RainToday' : 'No'}

predict_input(single_input)
# %%
# Saving and Loading Trained Models
import joblib

#Let's first create a dictionary containing all the required objects
aussie_rain = {
    'model' : model,
    'imputer' : imputer,
    'scaler' : scaler,
    'encoder' : encoder,
    'input_cols' : input_cols,
    'target_col' : target_col,
    'numeric_cols' : numeric_cols,
    'categorical_cols' : categorical_cols,
    'encoded_cols' : encoded_cols,
}

# We can now save this dictionary to disk using joblib
joblib.dump(aussie_rain, 'aussie_rain_model.joblib')

# %%
# We can load the model back using joblib.load
aussie_rain2 = joblib.load('aussie_rain_model.joblib')

#Let's use the loaded model to make predictions on the original test set
test_preds2 = aussie_rain2['model'].predict(X_test)
accuracy_score(test_targets, test_preds2)
