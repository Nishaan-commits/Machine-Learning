# %%
from urllib.request import urlretrieve
from IPython.display import display
import pandas as pd




medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
urlretrieve(medical_charges_url, 'medical.csv')

medical_df  = pd.read_csv('medical.csv')
display(medical_df) 
# %%
