# %%
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load Data
data = pd.read_csv('weatherAUS.csv')
data.dropna(subset=['RainToday','RainTomorrow'], inplace=True)

# Create Training, Validation and Test Sets
year = pd.to_datetime(data['Date']).dt.year
train_df, val_df, test_df = data[year < 2015], data[year == 2015], data[year > 2015]

# Create inputs and Targets 
input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'
train_inputs, train_targets = train_df[input_cols].copy(), train_df[target_col].copy()
val_inputs, val_targets = val_df[input_cols].copy(), val_df[target_col].copy()
test_inputs, test_targets = test_df[input_cols].copy(), test_df[target_col].copy

# Identify Numerical and Categorical Columns
numeric_cols = train_inputs.select_dtypes(include=['number']).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

# Impute Missing Values
imputer = SimpleImputer(strategy='mean').fit(data[numeric_cols])
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

# Scale numerical Features
scaler = MinMaxScaler().fit(data[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# One hot encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(data[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

# Save processed data to disk
train_inputs.to_parquet('train_inputs.parquet')
val_inputs.to_parquet('val_inputs.parquet')
test_inputs.to_parquet('test_inputs.parquet')
pd.DataFrame(train_targets).to_parquet('train_targets.parquet')
pd.DataFrame(val_targets).to_parquet('val_targets.parquet')
pd.DataFrame(test_targets).to_parquet('test_targets.parquet')

# Load Processed Data from Disk
train_inputs = pd.read_parquet('train_inputs.parquet')
val_inputs = pd.read_parquet('val_inputs.parquet')
test_inputs = pd.read_parquet('test_inputs.parquet')
train_targets = pd.read_parquet('train_targets.parquet')[target_col]
val_targets = pd.read_parquet('val_targets.parquet')[target_col]
test_targets = pd.read_parquet('test_targets.parquet')[target_col]
# %%
# Model Training And Evaluation

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Select the columns to be used for training/prediction
X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

# Create and train the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, train_targets)

# Generate Predictions and probabilities
train_preds = model.predict(X_train)
train_probs = model.predict_proba(X_train)
accuracy_score(train_targets, train_preds)

# Helper function to predict, compute accuracy & plot confusion matrix
def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)
    acc = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(acc * 100))
    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    return preds

# Evaluate on Validation Set and Test Set
val_preds = predict_and_plot(X_val, val_targets, name='Validation Set')
test_preds = predict_and_plot(X_test, test_targets, name='Test Set')

# Save the trained model & load it back
aussie_rain = {'model': model, 'numeric_cols': numeric_cols, 'encoded_cols': encoded_cols,
               'imputer': imputer, 'scaler': scaler, 'encoder': encoder, 'input_cols': input_cols,
               'target_col': target_col, 'categorical_cols': categorical_cols}
joblib.dump(aussie_rain, 'aussie_rain_model.joblib')
aussie_rain2 = joblib.load('aussie_rain_model.joblib')
# %%
# Running These all at once may give error, it's just to summarize the workflow