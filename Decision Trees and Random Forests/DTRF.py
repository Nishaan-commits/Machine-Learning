# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
# %%

raw_df = pd.read_csv('weatherAUS.csv')
raw_df

# %%

# Let's drop any rows where the target column 'RainTomorrow' is missing
raw_df.dropna(subset=['RainTomorrow'],inplace=True)
# %%
raw_df.info()
# %%

# Preparing the data for training 
# We'll perform the following steps to prepare the dataset for training:
#1. Create a train/validation/test split
#2. Identify input and target columns
#3. Identify categorical and numerical columns
#4. Impute missing numerical values
#5. Scale numeric values to the range [0,1]
#6. One-hot encode categorical columns

plt.title('No. of rows per year')
sns.countplot(x=pd.to_datetime(raw_df.Date).dt.year)
# %%
# While working with Chronological data, it's often a good idea to seperate the 
# training , validation and test sets based on time.

year  = pd.to_datetime(raw_df.Date).dt.year

train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]
print('Train_df.shape: ', train_df.shape)
print('Val_df.shape: ', val_df.shape)
print('Test_df.shape: ', test_df.shape) 

# %%
# Input and Target Columns
# Excluding date and RainTomorrow for the input columns
input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

# Numerical and Categorical Columns
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes(include='object').columns.tolist()

#Imputing Missing Numerical Values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean').fit(raw_df[numeric_cols])

train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

# Exercise: Try a different imputation strategy and see how it affects model performance

# Scaling Numerical Values to the range [0,1]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(raw_df[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

#Exercise: Try a different scaling strategy and observe how it affects the results

# One-Hot Encoding Categorical Columns
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(raw_df[categorical_cols])

encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

# Exercise: Try a different encoding strategy and observe how it affects the results

# Let's drop the textual categorical columns, so that we're left with just numeric data

X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

# %%
# Training and Visualizing Decision Trees
# A Decision Tree in machine learning works in exactly the same way, except that we let 
# the computer figure out the optimal structure & hierarchy of decisions, instead of
# coming up manually.


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state = 42)
model.fit(X_train, train_targets) 

# %%
# Evaluation of our model

from sklearn.metrics import accuracy_score, confusion_matrix
train_preds = model.predict(X_train)
pd.Series(train_preds).value_counts()

# %%
# The Decision trees also returns probabilities for each prediction
train_probs = model.predict_proba(X_test)
train_probs

# Damn decision tree is so confident about its predictions!
# %%
accuracy_score(train_preds, train_targets)
# As the training set accuracy is close to 100% we must test it for validation and test sets

# %%
# We can make Predictions and compute accuracy in one step using model.score
model.score(X_val, val_targets)

# Although the training accuracy is 100%, the accuracy on the validation set is only 
# 79%, which is marginally better than always prediction 'no'

# %%
val_targets.value_counts() / len(val_targets)

# It appears that the model has learned the training examples perfect, and doesn't 
# generalize well to unseen data. This phenomenon is called overfitting, and reducing
# overfitting is one of the most important parts of any machine learning project.

# %%
# Visualization
# We can visualize the decision tree learned from the training_data

from sklearn.tree import plot_tree, export_text

plt.figure(figsize=(80,20))
plot_tree(model, feature_names=X_train.columns, max_depth=2, filled=True)

# Can you see how the model classifies a given input as a Series of decisions? The tree is
# truncated here, but following any path from the root node to a leaf will result in "Yes" 
# or "No". Do you see how a decision tree differs from a logistic regression model?

# %%

# How a Decision Tree is created 
# Note the gini value in each box. this is the loss function used by the decision tree to decide
# which column should be used for splitting the data, and at what point the column should be
# split. A lower gini index indicates a better split. A perfect split (only one class on each side) 
# has a gini index of 0.

# Let's check the depth of the tree created.
model.tree_.max_depth

# %%

# We can also display the tree as text, which can be easier to follow for deeper trees
tree_text = export_text(model, max_depth=10, feature_names=list(X_train.columns))
print(tree_text[:5000])

# %%
# Feature Importance
# Based on the gini index computations, a decision tree assigns an "importance" value to each 
# feature. These values can be used to interpret the results given by a decision tree.

model.feature_importances_

# %%
# Let's turn into a dataframe and visualize the most important features

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

importance_df.head(10)

# %%
plt.title('Feature Importance')
sns.barplot(data=importance_df.head(10), x='importance', y='feature')
# %%
# Hyperparameter Tuning and Overfitting

# The process of reducing overfitting is known as regularization.
# The DecisionTreeClassifier accepts several arguments, some of which can be modified to reduce overfitting.
?DecisionTreeClassifier

# %%
# These arguments are called hyperparameters because they must be configured manually as opposed to parameters
# within the model which are learned from the data. We'll explore a couple of hyperparameters:
#1 max_depth
#2 max_leaf_nodes

# %%
# By reducing the maximum depth of the decision tree, we can prevent the tree from memorizing all training
# examples, which may lead to better generalization

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, train_targets)

# %%
# Let's check the accuracy on the training and validation sets
model.score(X_train, train_targets)
# %%
model.score(X_val, val_targets)
# %%
plt.figure(figsize=(80,20))
plot_tree(model, feature_names=X_train.columns, filled=True, rounded=True, class_names = model.classes_)
# %%
# Let's experiment with different depths using a helper function

def max_depth_error(md):
    model = DecisionTreeClassifier(max_depth=md, random_state=42)
    model.fit(X_train, train_targets)
    train_error = 1 - model.score(X_train, train_targets)
    val_error = 1 - model.score(X_val, val_targets)
    return {'Max Depth': md, 'Training Error': train_error, 'Validation Error': val_error}

errors_df = pd.DataFrame([max_depth_error(md) for md in range(1,21)])
errors_df
# %%
# Common Pattern you'll see with all Machine learning algorithms
plt.figure()
plt.plot(errors_df['Max Depth'], errors_df['Training Error'])
plt.plot(errors_df['Max Depth'], errors_df['Validation Error'])
plt.title('Training vs Validation Error')
plt.xticks(range(0,21,2))
plt.xlabel('Max Depth.')
plt.ylabel('Prediction Error (1-Accuracy)')
plt.legend(['Training', 'Validation'])

# %%
#  You'll often need to tune hyperparameters carefully to find the optimal fit. In the above case, it appears
# that a maximum depth of 7 results in the lowest validation error.

model = DecisionTreeClassifier(max_depth = 7, random_state=42)
model.fit(X_train, train_targets)
model.score(X_train, train_targets)

# %%
# Max_leaf_nodes

model = DecisionTreeClassifier(max_leaf_nodes=128, random_state=42).fit(X_train, train_targets)
model.score(X_train, train_targets)
# %%
model.score(X_val, val_targets)

# %%
model.tree_.max_depth

# Notice that the model was able to achieve a greater depth of 12 for certain paths while keeping other paths shorter.
model_text = export_text(model, feature_names =list(X_train.columns))
print(model_text[:3000])
# %%
# Exercise: Find the combination of max_depth and max_leaf_nodes that results in the highest validation accuracy.

# Exercise: Explore and experiment with other arguments of DecisionTree. Refer to the docs for details:
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

# Exercise: A more advanced technique (but less commonly used technique) for reducing overfitting in decision
# trees is known as cost-complexity pruning. Learn more about it here:
# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html. Implement cost
# complexity pruning. Do you see any improvement  in the validation accuracy?

# %%
# Training a Random Forest
# While tuning the hyperparameters of a single decision tree may lead to some improvements, a much more 
# effective strategy is to combine the results of several decision trees trained with slightly different 
# parameters. This is called a random forest model.

# The key idea here is that each decision tree in the forest will make different kinds of errors, and upon
# averaging, many of their errors will cancel out. This idea is also commonly known as the "wisdom of the crowd".

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1, random_state=42)

# n_jobs allows the random forest to use multiple parallel workers to train decision trees
model.fit(X_train, train_targets)
model.score(X_train, train_targets)
# %%
model.score(X_val, val_targets)
# %%
# This general technique of combining the results of many models is called "ensembling", it works because
# most errors of individual models cancel out on averaging.

# We can also look at the probabilities for the predictions.The probability of a class is simply the fraction
# of trees which that predicted the given class

train_probs = model.predict_proba(X_train)
train_probs
# %%
# We can access individual decision trees using model.estimators_
len(model.estimators_)
# %%
# Exercise: Verify that none of the individual decision trees have a better validation accuracy than the
# random forest.


# %%
# Just like decision trees, random forests also assign an "importance" to eahc feature, by combining the importance
# values from inidividual trees

importance_df = pd.DataFrame({
    'features': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending = False)

importance_df.head(10)
# %%
plt.title('Feature Importance')
sns.barplot(data=importance_df.head(5), x = 'importance', y='features')
# The distribution is a lot less skewed than decision trees
# %%
# Hyperparameter Tuning with Random Forests
# Learn more at https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

?RandomForestClassifier


# %%
# Let's create a base model with which we can compare models with tuned hyperparameters
base_model = RandomForestClassifier(random_state = 42, n_jobs=-1).fit(X_train, train_targets)
base_train_acc =  base_model.score(X_train,train_targets)
base_val_acc = base_model.score(X_val, val_targets)
# %%
base_accs = base_train_acc, base_val_acc
base_accs
# %%
# n_estimators  
# This argument controls the number of Decision trees in the Random forest. The default value is 100. For 
# larger datasets, it helps to have a greater number of estimators. As a general rule, try to have as feww
# as needed
# Let's try 10 estimators
model = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators = 10).fit(X_train, train_targets)
model.score(X_train, train_targets), model.score(X_val, val_targets)

# %%
model =RandomForestClassifier(random_state = 42, n_jobs = -1, n_estimators=500).fit(X_train, train_targets)
model.score(X_train, train_targets), model.score(X_val, val_targets)
# Just a 0.13% increase in accuracy so we may?may not need this much kinda depends upon the circumstances
# %%
# Exercise: Vary the value of n_estimators and plot the graph between training error and validation error.
# What is the optimal value of n_estimators?
# Let's find out, need a helper function for this

def optimal_esti(md):
    model = RandomForestClassifier(random_state = 42, n_jobs = -1, n_estimators=md).fit(X_train, train_targets)
    train_acc = model.score(X_train, train_targets)
    val_acc = model.score(X_val, val_targets)
    return {'Estimators ': md, 'Training Accuracy': train_acc, 'Validation Accuracy': val_acc }

estimators = pd.DataFrame([optimal_esti(md) for md in range(100, 201)])
estimators.sort_values('Validation Accuracy', ascending=False).head(10)
# %%
plt.title('Estimators vs. Accuracy')
sns.barplot(data=estimators.sort_values('Validation Accuracy', ascending=False).head(10), x='Estimators ', y='Validation Accuracy')

# %%
# max_depth and max_leaf_nodes
# These arguments are passed directly to each decision tree, and control the maximum depth and max no. of 
# leaf nodes of each tree. 

# Let's define a helper function to make it easy to test hyperparameters
def test_params(**params):
    model = RandomForestClassifier(random_state =42, n_jobs = -1, **params).fit(X_train, train_targets)
    return model.score(X_train, train_targets),model.score(X_val, val_targets)

# %%
# Let's test a few values of max_depth and max_leaf_nodes
test_params(max_depth=5)

# %%
test_params(max_depth=26)
# %%
test_params(max_leaf_nodes=2**5)
# %%
# Exercise : Vary the value of max_depth and plot the graph between training error and validation error.
# What is the optimal value of max_depth? Do the same for max_leaf_nodes.

depth_data = pd.DataFrame([
    test_params(max_depth=md) for md in range(1,50,2)])
depth_data['Max Depth'] = range(1,50,2)
# %%
plt.figure()
plt.plot(depth_data['Max Depth'], depth_data['Training error'])
plt.plot(depth_data['Max Depth'], depth_data['Validation error'])
plt.title('Training vs Validation Error')
plt.xlabel('Max Depth')
plt.ylabel('Prediction Error (1-Accuracy)')
plt.legend(['Training','Validation'])

# %%
depth_data.sort_values('Validation error').head(10)
# Optimal depth = 39
# Optimal leaf nodes = ??

# %%
# max_features 
# Instead of picking all features (columns) for every split, we can specify that only a fraction of features
# be chosen randomly to figure out a split.

# max_features : {"auto","sqrt","log2"},int or float, default="auto" 
# The number of features to consider when looking for the best split:

# If int, then consider max_features features at each split.
# If float, then max_features is a fraction and round(max_features * n_features) features are considered
# at each split.
# If "auto" or "sqrt", then max_features=sqrt(n_features)
# If "log2". then max_features=log2(n_features)

# Notice that the default value auto causes only root n out of totoal features(n) to be chosen randomly at
# each split. This is the reason each decision tree in the forest is different. while it may seem counterintuitive
# choosing all features for every split of every tree will lead to identical trees, so the random forest 
# will not generalize well

test_params(max_features='log2')


# %%
test_params(max_features=3)
# %%
test_params(max_features=6)
# %%
# Exercise: Find the optimal value of max_features for this dataset

# %%
# min_samples_split and min_samples_leaf
# By default, the dtc tries to split every node that has 2 or more. You can increase the values of these 
# arguments to change this behavior and reduce overfitting, especially for very large datasets.

test_params(min_samples_split=3,min_samples_leaf=2)
# %%
test_params(min_samples_split=5, min_samples_leaf=3)

# %%
# min_impurity_decrease 
# This argument is used to control the threshold for splitting nodes. A node will be split if this split
# induces a decrease of the impurity (Gini index) greater than or equal to this value.

test_params(min_impurity_decrease=1e-7)
# %%
test_params(min_impurity_decrease=1e-9)
# %%
base_accs
# %%
# bootstrap, max_samples
# By default, a random forest doesn't use the entire dataset for training each decision tree. Instead it
# applies a technique called bootstrapping. For each tree, rows from the dataset are picked one by one 
# randomly, with replacement i.e. some rows may not show up at all, while some rows may show up multiple
# times. By default it's True.

test_params(bootstrap=False)


# %%
# You can control the no. of fraction of rows to be considered for each bootstrap using max_samples.
test_params(max_samples=0.95)
# %%
# class_weight
test_params(class_weight={'No': 1, 'Yes': 2})
# %%
# Putting it together
# Let's train a random forest with customized hyperparameters based on our learnings.
test_params(n_estimators=101,max_features=7,max_depth=30,class_weight={'No': 1, 'Yes': 2}) 

# %%
