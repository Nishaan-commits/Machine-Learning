# Exploratory Data Analysis

# %%
from urllib.request import urlretrieve
import pandas as pd 
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns




medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
urlretrieve(medical_charges_url, 'medical.csv')

medical_df  = pd.read_csv('medical.csv')

# %%
medical_df.info()
medical_df.describe()
# %%
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10,6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
# %%
fig = px.histogram(medical_df,
                   x = 'age',
                   marginal = 'box',
                   nbins=47,
                   title='Distribution of Age')
fig.update_layout(bargap=0.1)
fig.show()
# %%

fig = px.histogram(medical_df,
                    x='bmi',
                    marginal='box',
                    color_discrete_sequence=['red'],
                    title = 'Distribution of BMI (Body Mass Index)')
fig.update_layout(bargap=0.1)
fig.show()
# %%
fig = px.histogram(medical_df,
                   x ='charges',
                   marginal='box',
                   color='sex',
                   color_discrete_sequence=['green','grey'],
                   title = 'Annual Medical Charges')
fig.update_layout(bargap=0.1)
fig.show()
# %%
medical_df.smoker.value_counts()
px.histogram(medical_df, x='smoker', color='children', title='Smoker')
# %%
fig = px.scatter(medical_df,
                 x = 'age',
                 y = 'charges',
                 color = 'smoker',
                 opacity = 0.8,
                 hover_data=['sex'],
                 title = 'Age vs. Charges')
fig.update_traces(marker_size=5)
fig.show()
# %%
fig = px.scatter(medical_df, 
                 x = 'children',
                 y = 'charges',
                 color = 'smoker',
                 opacity=0.8,
                 hover_data=['sex'],
                 title = 'BMI vs. Charges')
fig.update_traces(marker_size=5)
fig.show()
# %%
fig = px.violin(medical_df,y ='charges', x ='children')
fig.show()
# %%
medical_df.charges.corr(medical_df.age)
# %%
medical_df.charges.corr(medical_df.bmi)

# %%
# To compute the correlation for categorical columns, they must first be converted into numeric columns.
smoker_values = {'no' : 0, 'yes': 1}
smoker_numeric = medical_df.smoker.map(smoker_values)
medical_df.charges.corr(smoker_numeric)
# %%
numeric_df = medical_df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), cmap='Reds', annot = True)
plt.title('Correlation Matrix')
# %%
# Linear Regression using a single feature

non_smoker_df = medical_df[medical_df.smoker == 'no']
plt.title('Age vs Charges')
sns.scatterplot(data = non_smoker_df, x = 'age', y='charges', alpha=0.7, s=15)
# %%
# In the above case, the x axis shows "age" and the y axis shows "charges". Thus, we assume the following relationship between the two:

# charges = w×age+b

# We'll try determine w and b for the line that best fits the data.

#     This technique is called linear regression, and we call the above equation a linear regression model, because it models the relationship between "age" and "charges" as a straight line.

#     The numbers w and b are called the parameters or weights of the model.

#     The values in the "age" column of the dataset are called the inputs to the model and the values in the charges column are called "targets".

def estimate_charges(age, w, b): # Our very first model
    return w * age + b

w=50
b=100
ages = non_smoker_df.age
estimate_charges = estimate_charges(ages,w,b)

plt.plot(ages, estimate_charges, 'r-')
plt.xlabel('Age')
plt.ylabel('Estimated Charges')
# %%
# As Expected, the points lie on a straight line 
# We can overlay this line on the actual data, so see how well our model fits the data

target = non_smoker_df.charges

plt.plot(ages, estimate_charges, 'r', alpha=0.8)

plt.scatter(ages, target, s=8, alpha=0.8)
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend(['Estimate', 'Actual'])
# %%
def try_parameters(w,b):
    ages = non_smoker_df.age 
    target = non_smoker_df.charges

    predictions = estimate_charges(ages, w, b)

    plt.plot(ages, predictions, 'r', alpha=0.9)
    plt.scatter(ages, target, s=8, alpha=0.8)
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend(['Estimate', 'Actual'])

try_parameters(290,-4000)
# %%
# Computing Root Mean Squared Error for our model with a sample set of weights
import numpy as np

def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))

w = 300
b = -4000
try_parameters(w,b)

targets = non_smoker_df['charges']
predicted = estimate_charges(non_smoker_df.age,w,b)

rmse(targets, predicted)
# The result is the loss because it indicates how bad the model is at predicting the target variables.
# %%
# Modifying try_parameters to desplay the loss as well

def try_parameters(w, b):
    ages = non_smoker_df.age 
    target = non_smoker_df.charges
    predictions = estimate_charges(ages, w, b)

    plt.plot(ages, predictions, 'r', alpha=0.9)
    plt.scatter(ages, target, s=8, alpha=0.8)
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend(['Prediction', 'Actual'])

    loss = rmse(target, predictions)
    print("RMSE Loss: ", loss)

try_parameters(299,-3342)
# %%
# Scikit-Learn
from sklearn.linear_model import LinearRegression

# Model Object
model = LinearRegression()
help(model.fit)

# Note that the input x must be a 2d-array, so we'll need  to pass a dataframe, instead of a single column
# %%
inputs = non_smoker_df[['age']]
targets = non_smoker_df.charges
print('inputs.shape :', inputs.shape)
print('targets.shape :',targets.shape)
# %%
model.fit(inputs, targets)
model.predict(np.array([[23],
                       [37],
                       [61]]))
# %%
predictions = model.predict(inputs)
predictions
# %%
rmse(targets, predictions)
# %%
# The parameters of the model are stored in the coef_ and intercept_ properties
# w
model.coef_
# %%
# b
model.intercept_
# %%
try_parameters(model.coef_, model.intercept_)
# %%
from sklearn.linear_model import SGDRegressor

model = SGDRegressor()
model.fit(inputs, targets)
predictions = model.predict(inputs)
# %%
rmse(targets, predictions)
# %%
model.coef_
# %%
model.intercept_
# %%
try_parameters(model.coef_, model.intercept_)
# %%

# Doing the same for Smokers now [LinearRegression Model]
from sklearn.linear_model import LinearRegression
model = LinearRegression()
smoker_df = medical_df[medical_df.smoker == 'yes']
inputs = smoker_df[['age']]
targets = smoker_df.charges
print('inputs.shape :', inputs.shape)
print('targets.shape :', targets.shape)
model.fit(inputs, targets)

# %%
predictions = model.predict(inputs)
rmse(targets, predictions)

# %%
model.coef_

# %%
model.intercept_

# %%
# need to update try_parameters for smokers
def try_parameters(w, b):
    ages = smoker_df.age 
    target = smoker_df.charges
    predictions = estimate_charges(ages, w, b)

    plt.plot(ages, predictions, 'r', alpha=0.9)
    plt.scatter(ages, target, s=8, alpha=0.8)
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend(['Prediction', 'Actual'])

    loss = rmse(target, predictions)
    print("RMSE Loss: ", loss)

try_parameters(model.coef_, model.intercept_)

# %%
# Let's Try the SGDRegressor for smokers now
# The parameters stuff is from chatgpt it felt nice so i added it 
from sklearn.linear_model import SGDRegressor
model = SGDRegressor(
    max_iter = 16000, #more epochs -> more chance to reach optimal
    tol = 1e-4, #stop when improvements are very small
    eta0 = 0.001, # learning rate
    random_state = 42, # fixes randomness so results are reproducible
    shuffle = False # shuffle data every epoch(usually good -chatgpt)
)

model.fit(inputs, targets)
predictions = model.predict(inputs)
rmse(targets, predictions)

# %% 
model.coef_

# %% 
model.intercept_

# %% 
try_parameters(model.coef_, model.intercept_)

# %%
# Before moving out to next topic, Summarizing what we did so far
#1 create inputs and targets
inputs, targets = non_smoker_df[['age']], non_smoker_df['charges']

#2 create and train the model
model = LinearRegression().fit(inputs, targets)

#3 Generate predictions
predictions = model.predict(inputs)

#4 Compute loss to evaluate the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# %%
# Linear Regression with multiple features, age and bmi to predict charges
# charges = w1 * age + w2 * bmi + b
#1 create inputs and targets
inputs, targets = non_smoker_df[['age','bmi']], non_smoker_df['charges']

#2 create and train the model
model = LinearRegression().fit(inputs, targets)

#3 Generate predictions
predictions = model.predict(inputs)

#4 Compute loss to evaluate the model
loss = rmse(targets, predictions)
print('Loss:', loss)

# %%
# the loss didnt reduce much because bmi has very weak correlation with charges, especially for non smokers
non_smoker_df.charges.corr(non_smoker_df.bmi)

# %%
# Going One step further by adding the final numeric column: children
# charges = w1 * age + w2 * bmi + w3 * children + b
non_smoker_df.charges.corr(non_smoker_df.children)

fig = px.strip(non_smoker_df, x='children', y='charges', title = 'Children vs Charges')
fig.update_traces(marker_size=4, marker_opacity=0.7)
fig.show()

# %%
#1 create inputs and targets
inputs, targets = non_smoker_df[['age','bmi', 'children']], non_smoker_df['charges']

#2 create and train the model
model = LinearRegression().fit(inputs, targets)

#3 Generate predictions
predictions = model.predict(inputs)

#4 Compute loss to evaluate the model
loss = rmse(targets, predictions)
print('Loss:', loss)

# %%
# Doing the same For smokers now
# Let's check out the correlations first
numeric_df = smoker_df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), cmap='Reds', annot = True)
plt.title('Correlation Matrix')

# %%
#1 create inputs and targets
inputs, targets = smoker_df[['age','bmi', 'children']], smoker_df['charges']

#2 create and train the model
model = LinearRegression().fit(inputs, targets)

#3 Generate predictions
predictions = model.predict(inputs)

#4 Compute loss to evaluate the model
loss = rmse(targets, predictions)
print('Loss:', loss)

# %%
# Let's consider the entire  dataset now
#1 create inputs and targets
inputs, targets = medical_df[['age','bmi', 'children']], medical_df['charges']

#2 create and train the model
model = LinearRegression().fit(inputs, targets)

#3 Generate predictions
predictions = model.predict(inputs)

#4 Compute loss to evaluate the model
loss = rmse(targets, predictions)
print('Loss:', loss)

# %%
# As the loss is way too much for the complete dataset we are going to use categorical columns as well from now on 

# To use the categorical columns, we simply need to convert them to numbers. 3 ways to do this
#1 If a categorical column has just two categories (binary category), then we can replace their values to 0 and 1
#2 If a categorical column has more than 2 categories, we can perform one-hot encoding i.e. create a new column for each category with 1s and 0s
#3 If the categories have natural order(e.g. cold,neutral,warm,hot), then they can be converted to numbers(1,2,3,4) preserving the order. These are called ordinals

# Binary Categories
# Going to convert smoker column as it just has two values "yes" and "no"
sns.barplot(data = medical_df, x = 'smoker', y = 'charges')

# %%
# Creating a new column smoker_code
smoker_codes = {'no' : 0, 'yes' : 1}
medical_df['smoker_code'] = medical_df.smoker.map(smoker_codes)
medical_df.charges.corr(medical_df.smoker_code)

# %%
# We can now use the smoker_code column for linear Regression
# Let's consider the entire  dataset now
#1 create inputs and targets
inputs, targets = medical_df[['age','bmi', 'children','smoker_code']], medical_df['charges']

#2 create and train the model
model = LinearRegression().fit(inputs, targets)

#3 Generate predictions
predictions = model.predict(inputs)

#4 Compute loss to evaluate the model
loss = rmse(targets, predictions)
print('Loss:', loss)

# %%
# As we reduced our loss so much by just including one categorical column it's time to involve the sex column as well 
# First let's analyze the correlation
sns.barplot(data = medical_df, x = 'sex', y = 'charges')

sex_codes = { 'female' : 0, 'male' : 1}
medical_df['sex_code'] = medical_df.sex.map(sex_codes)
medical_df.charges.corr(medical_df.sex_code)

# %%
# Using the sex_code column for our model now
#1 create inputs and targets
inputs, targets = medical_df[['age','bmi', 'children','smoker_code','sex_code']], medical_df['charges']

#2 create and train the model
model = LinearRegression().fit(inputs, targets)

#3 Generate predictions
predictions = model.predict(inputs)

#4 Compute loss to evaluate the model
loss = rmse(targets, predictions)
print('Loss:', loss)
# %%

# One Hot Encoding
# The region column contains 4 values, so we will have to use one hot encoding and create a new column for each region

sns.barplot(data = medical_df, x = 'region', y = 'charges')
# %%
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit(medical_df[['region']])
enc.categories_
# %%
one_hot = enc.transform(medical_df[['region']]).toarray()
one_hot
# %%
medical_df[['northeast','northwest','southeast','southwest']] = one_hot
medical_df
# %%
# Using The One hot Regions now
#1 create inputs and targets
input_cols = ['age','bmi','children','smoker_code','sex_code','northeast','northwest','southeast','southwest']
inputs, targets = medical_df[input_cols], medical_df['charges']

#2 create and train the model
model = LinearRegression().fit(inputs, targets)

#3 Generate predictions
predictions = model.predict(inputs)

#4 Compute loss to evaluate the model
loss = rmse(targets, predictions)
print('Loss:', loss)
# %%
# Exercise: Is One LR model better or 2 seperate LR models for smokers and non smokers Let's find out
#1 create inputs and targets
smoker_df = medical_df[medical_df.smoker_code == 1]
non_smoker_df = medical_df[medical_df.smoker_code == 0]
input_cols = ['age','bmi','children','sex_code','northeast','northwest','southeast','southwest']
inputs, targets = smoker_df[input_cols], smoker_df['charges']
inputs1, targets1 = non_smoker_df[input_cols], non_smoker_df['charges']

#2 create and train the model
model = LinearRegression().fit(inputs, targets)
model1 = LinearRegression().fit(inputs1, targets1)

#3 Generate predictions
predictions = model.predict(inputs)
predictions1 = model1.predict(inputs1)

#4 Compute loss to evaluate the model
loss = rmse(targets, predictions)
loss1 = rmse(targets1, predictions1)
print('Loss:', loss)
print('Loss1:', loss1)

# %%
model.coef_
# As we see the bmi and northeast has higher weight than age which is contradictory we run into this issue because of each column having different range of values
# To solve this issue we have something called as Feature Scaling
# In which we scale values of each numeric column by subtracting the mean and dividing by standard deviation
# We can apply scaling using the StandardScaler class from scikit-learn

# %%
from sklearn.preprocessing import StandardScaler

numeric_cols = ['age','bmi','children']
scaler = StandardScaler()
scaler.fit(medical_df[numeric_cols])
# %%
scaler.mean_
# %%
scaler.var_
# %%
scaled_inputs = scaler.transform(medical_df[numeric_cols])
scaled_inputs
# %%
# Combining with categorical data
cat_cols = ['smoker_code','sex_code','northeast','northwest','southeast','southwest']
categorical_data = medical_df[cat_cols].values

# %%
inputs = np.concatenate((scaled_inputs, categorical_data), axis=1)
targets = medical_df.charges

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate Predictions
predictions = model.predict(inputs)

# Compute loss to evaluate the model
loss = rmse(targets, predictions)
print('loss: ', loss)

# %%
# We can now compare the weights in the linear regression formula
weights_df = pd.DataFrame({
    'feature' : np.append(numeric_cols + cat_cols, 1),
    'weight' : np.append(model.coef_, model.intercept_)
})
weights_df.sort_values('weight', ascending = False)
# %%
# Creating a Test Set
from sklearn.model_selection import train_test_split
inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size = 0.1)

model = LinearRegression().fit(inputs_train, targets_train)
predictions_test = model.predict(inputs_test)
loss = rmse(targets_test, predictions_test)
print("Test Loss: ", loss)

# %%
# Comparing with the training loss
predictions_train = model.predict(inputs_train)

loss = rmse(targets_train, predictions_train)
print('Training loss: ', loss)
