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
