# %%
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10,6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

# %%
ross_df = pd.read_csv('train.csv', low_memory=False)
store_df = pd.read_csv('store.csv')
test_df = pd.read_csv('test.csv')
submission_df = pd.read_csv('sample_submission.csv')

# %%
submission_df

# %%
# First step: Let's merge the information from store_df to train_df and test_df
merged_df = ross_df.merge(store_df, how='left', on='Store')
merged_test_df = test_df.merge(store_df, how='left', on='Store')

# %%
merged_df
# %%
# Exercise: Perform exploratory data analysis and visualization on the dataset. Study the 
# distribution of values in each column, and their relationship with the target column Sales.

fig = px.scatter(merged_df,
                 x='Sales',
                 y='Customers',
                 color = 'Promo',
                 opacity = 0.8,
                 title = 'Sales vs Customers')
fig.update_traces(marker_size=5)
fig.show()
# %%
# Correlation Matrix
numeric_df = ross_df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), cmap='Reds', annot = True)
plt.title('Correlation Matrix')
# Seems like Customers, Open and Promo are the most important factors from numerical categories

# %%
# Preprocessing and Feature Engineering
# Date

# First let's convert Date to a datecolumn and extract parts of the date
def split_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['WeekOfYear'] = df.Date.dt.isocalendar().week
split_date(merged_df)
split_date(merged_test_df)

# %%
merged_df
# %%
merged_df.Sales.corr(merged_df.WeekOfYear)
# %%
# Store Open/Closed
# The sales are zero when the store is closed
# So we can hardcode it and remove the rows
merged_df[merged_df.Open == 0].Sales.value_counts()
merged_df = merged_df[merged_df.Open == 1].copy()

# %%
# Competitions
# We can use the columns CompetitionOpenSince[Month/Year] from store_df to compute the number
# of months for which a competitor has been open near the store.

def comp_months(df):
    df['CompetitionOpen'] = 12 * (df.Year - df.CompetitionOpenSinceYear) + (df.Month - df.CompetitionOpenSinceMonth)
    df['CompetitionOpen'] = df['CompetitionOpen'].map(lambda X: 0 if X<0 else X).fillna(0)

comp_months(merged_df)
comp_months(merged_test_df)

merged_df

# %%
# Additional Promotion
# We can also add additional columns to indicate how long a store has been running Promo2 and
# whether a new round of Promo2 starts in the current month.

def check_promo_month(row):
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep',
                 10:'Oct', 11:'Nov', 12:'Dec'} 
    try:
        months = (row['PromoInterval'] or '').split(',')
        if row['Promo2Open'] and month2str[row['Month']] in months:
            return 1
        else: 
            return 0
    except Exception:
        return 0

def promo_cols(df):
    # Month since  Promo2 was open
    df['Promo2Open'] = 12 * (df.Year - df.Promo2SinceYear) + (df.WeekOfYear - df.Promo2SinceWeek)*7/30.5
    df['Promo2Open'] = df['Promo2Open'].map(lambda x: 0 if x < 0 else x).fillna(0) * df['Promo2']

    # Whether a new round of promotions was started in the current month
    df['IsPromo2Month'] = df.apply(check_promo_month, axis=1) * df['Promo2']

promo_cols(merged_df)
promo_cols(merged_test_df)
# %%
merged_df
# The features related to competition and promotion are now much more useful
# Feature engineering is a lot of back and forth, here we are just getting the idea of how people
# do things but in reality u do some feature engineering check with EDA you may not like the
# results u delete them then make more fresh columns until u see some correlation or trend
# I would suggest myself doing this on Titanic Survival Prediction if i forget to practice 

# %%
# Input and Target Columns
# Let's select the columns for training

merged_df.columns
# %%
input_cols = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
              'CompetitionDistance','CompetitionOpen','Day','Month','Year','WeekOfYear','Promo2',
              'Promo2Open', 'IsPromo2Month']
target_col = 'Sales'
inputs = merged_df[input_cols].copy()
targets = merged_df[target_col].copy()
test_inputs = merged_test_df[input_cols].copy()

# %%
# Let's identify numeric and categorical columns. Note that we can treat binary categorical
# columns (0/1) as numeric columns.

numeric_cols = ['Store', 'Promo', 'SchoolHoliday','CompetitionDistance', 'CompetitionOpen', 'Promo2',
                'Promo2Open', 'IsPromo2Month', 'Day', 'Month', 'Year', 'WeekOfYear']
categorical_cols = ['DayOfWeek','StateHoliday','StoreType','Assortment']

# %%
# Impute missing numeric columns
inputs[numeric_cols].isna().sum()
# Seems like Competition Distance is the only missing value, and we can simply fill it with the
# highest value(to indicate competition is very far away)

# %%
max_distance = inputs.CompetitionDistance.max()
inputs.fillna({'CompetitionDistance': max_distance}, inplace=True)
test_inputs.fillna({'CompetitionDistance': max_distance}, inplace=True)

# %%
# Scale Numeric Values
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().fit(inputs[numeric_cols])
inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# %%
# Encode Categorical Columns
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(inputs[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

inputs[encoded_cols] = encoder.transform(inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

# %%
# Finally, let's extract out all the numeric data for training

X = inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

# %%
# We haven't created a Validation set yet, because we'll use K-Fold Cross Validation
# Exercise: Look through the notebooks created by participants in the Kaggle competition and apply
# some feature engineering. https://www.kaggle.com

# %%
# Gradient Boosting
# We're now ready to train our gradient boosting machine(GBM) model. Here's how a GBM model works

#1 The average value of the target column and uses as an initial prediction every input.
#2 The residuals (difference) of the predictions with the targets are computed
#3 A decision tree of limited depth is trained to predict just the residuals of the input
#4 Predictions from the decision tree are scaled using a parameter called the learning rate (this prevents overfittin)
#5 Scaled Predictions from the tree are added to the previous predictions to obtain the new
#  and improved predictions
#6 Steps 2 to 5 are repeated to create new decision trees, each of which is trained to predict
#  just the residuals from the previous prediction

# The term "gradient" refers to the fact that each decision is trained with the purpose of reducing the 
# loss from the previous iteration(similar to gradient descent). The term "boosting" refers to the
# general technique of training the new models to improve the results of an existing model

# Exercise: Can you describe in your own words how a gradient boosting machine is different from
# a random forest?
# %%
# Training
from xgboost import XGBRegressor
model = XGBRegressor(random_state=42, n_jobs=-1, n_estimators=20, max_depth = 4).fit(X, targets)
# ?XGBRegressor

# %%
# Predictions
preds = model.predict(X)
preds

# %%
# Evaluation using RMSE
from sklearn.metrics import root_mean_squared_error

def rmse(a, b):
    return root_mean_squared_error(a, b)

rmse(preds, targets)

# %%
# Visualization
import matplotlib.pyplot as plt
from xgboost import plot_tree
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 30,30
# %%
plot_tree(model, rankdir='LR', tree_idx = 0)

# %%
plot_tree(model, rankdir='LR', tree_idx = 1)

# %%
plot_tree(model, rankdir='LR', tree_idx = 19)

# %%
# We can also visualize the tree as text.
trees = model.get_booster().get_dump()

# %%
len(trees)

# %%
print(trees[0])

# %%
# Feature Importance
importance_df = pd.DataFrame({
    'feature' : X.columns,
    'importance' : model.feature_importances_
}).sort_values('importance', ascending=False)

# %%
importance_df.head(10)

# %%
import seaborn as sns
plt.figure(figsize=(10,6))
plt.title('Feature Importance')
sns.barplot(data=importance_df.head(10), x='importance', y='feature')

# %%
# K-Fold Cross Validation
# We split the training dataset into K splits and each time one split is taken as validation set for the
# rest.
# It's not that good here because when it's time ordered data we should have validation data close to 
# test data like it should be in future wrt our training data, so only the last split will be useful.

from sklearn.model_selection import KFold

# %%
# Helper_function to return model, training and validation error
def train_and_evaluate(X_train, train_targets, X_val, val_targets, **params):
    model = XGBRegressor(random_state = 42, n_jobs=-1, **params)
    model.fit(X_train, train_targets)
    train_rmse = rmse(model.predict(X_train), train_targets)
    val_rmse = rmse(model.predict(X_val), val_targets)
    return model, train_rmse, val_rmse

# Now, we can use the KFold utility to create the different training/validations splits and train a 
# seperate model for each fold.
kfold = KFold(n_splits = 5)

# %%
models = []

for train_idxs, val_idxs in kfold.split(X):
    X_train, train_targets = X.iloc[train_idxs], targets.iloc[train_idxs]
    X_val, val_targets = X.iloc[val_idxs], targets.iloc[val_idxs]
    model, train_rmse, val_rmse = train_and_evaluate(X_train,
                                                     train_targets,
                                                     X_val,
                                                     val_targets,
                                                     max_depth=4,
                                                     n_estimators=20)
    models.append(model)
    print('Train RMSE: {}, Validation RMSE: {}'.format(train_rmse, val_rmse))

# %%
# Let's also define a function to average predictions from the 5 different models
import numpy as np

def predict_avg(models, input):
    return np.mean([model.predict(input) for model in models], axis = 0)

# %%
preds = predict_avg(models, X)
preds

# %%
# Hyperparameter Tuning and Regularization -> https://xgboost.readthedocs.io/en/latest/parameter.html

from sklearn.model_selection import train_test_split

X_train, X_val, train_targets, val_targets = train_test_split(X, targets, test_size=0.1)

def test_params(**params):
    model = XGBRegressor(n_jobs=-1, random_state=42, **params)
    model.fit(X_train, train_targets)
    train_rmse = rmse(model.predict(X_train), train_targets)
    val_rmse = rmse(model.predict(X_val), val_targets)
    print(f'Train RMSE: {train_rmse}, Validation RMSE: {val_rmse}')

# %%
# n_estimators -> No. of trees to be created. More trees = greater capacity
test_params(n_estimators=170)


# %%
# max_depth
# As you increase the max depth of each tree, the capacity of the tree increases and it can capture
# more information about the training set

for md in range(2,10):
    test_params(max_depth=md) 

# %%
# learning_rate
# The scaling factor to be applied to the prediction of each tree. A very high learning rate will lead
# to overfitting, and a low learning rate will lead to underfitting
test_params(n_estimators = 90, learning_rate=0.8)

# %%
# booster
# Instead of using Decision trees, XGBoost can also train a linear model for each iteration. This can be
# configured using booster

test_params(booster='gblinear')
# %%
# Exercise: Experiment with other hyperparameters like gamma, min_child_weight, max_delta_step, subsample
# colsample_bytree etc. and find their optimal values. 

# Exercise: Train a model with your best hyperparameters and evaluate its performance using 5-fold cross
# validation

# %%
# Putting it Together and Making Predictions
# Let's train a final model on the entire training set with custom hyperparameters.

model = XGBRegressor(n_jobs=-1, random_state=42, n_estimators=200,
                     learning_rate=0.7, max_depth=10, subsample=0.9,
                     colsample_bytree=0.7)
model.fit(X, targets)
# %%
preds = model.predict(X)
preds
# %%
rmse(preds, targets)
# %%
# Predicted Test inputs and then saved our results in submission_df
test_preds = model.predict(X_test)
submission_df['Sales'] = test_preds

# %%
submission_df.sample(10)

# %%
# Recall, however, if the store is not open, then the sales must be 0. Thus, wherever the value of Open
# in the test set is 0, we can set the sales to 0. Also, there some missing values for Open in the test
# set. We'll replace them with 1(open)

submission_df['Sales'] = submission_df['Sales'] * test_df.Open.fillna(1.)

# %%
# We can save the predictions as a CSV file.

submission_df.to_csv('submission.csv', index=None)

# %%
# Score he got: 0.13107
# Exercise: Experiment with different models and hyperparameters and try to beat the above 
# score. Take inspiration from the top notebooks on the "Code" tab of the competition.

# Exercise: Save the model and all the other required objects using joblib

# %%
# Write a function predict_input which can make predictions for a single input 
# provided as a dictionary. Make sure to include all the feature engineering and preprocessing
# steps.

sample_input = {
    'Store': 2,
    'DayOfWeek':4,
    'Promo': 1,
    'Date': '2015-09-30',
    'Open': 1,
    "StateHoliday": 'a',
    'SchoolHoliday': 1,
}

input_df = pd.DataFrame([sample_input])

# merge with stores_df
input_merged_df = input_df.merge(store_df, on='Store')

# Feature Engineering
# Dates
split_date(input_merged_df)
# Competition
comp_months(input_merged_df)
# Promo2
promo_cols(input_merged_df)
input_merged_df

# %%
inputs.head(1)

# %%
# Preprocessing
# Imputation - not required as CompDistance is present
# Scaling
input_merged_df[numeric_cols] = scaler.transform(input_merged_df[numeric_cols])
# Encoding
input_merged_df[encoded_cols] = encoder.transform(input_merged_df[categorical_cols])
# %%
# Selecting the right columns
X_input = input_merged_df[numeric_cols+encoded_cols]
# Pass it into the model
model.predict(X_input)[0]
# %%
sample_input
# %%
