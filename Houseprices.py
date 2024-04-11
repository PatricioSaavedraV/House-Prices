import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vRKrqz1DWLY6HhPKEhzXBGulCEHeGz04yOoAlxKtThPSqz1ks1BV4_L_eg4H0y8eZqLkJAySoEkSgsb/pub?output=csv')
raw_and_columns = 'Row and columns'
print(f"{raw_and_columns} {df.shape}")

df[['new_date', 'hour']] = df['date'].str.split(' ', expand=True)

df['new_date'] = pd.to_datetime(df['new_date'])
df['year'] = df['new_date'].dt.year
df['month'] = df['new_date'].dt.month
df['day'] = df['new_date'].dt.day

df['hour'] = pd.to_datetime(df['hour']).dt.time

df['year'] = df['year'].astype('int64')
df['month'] = df['month'].astype('int64')
df['day'] = df['day'].astype('int64')

df = df.drop(columns=['date'])
df = df.drop(columns=['new_date'])


print(df['year'].nunique())
print(df['month'].nunique())
print(df['day'].nunique())
print(df['hour'].nunique())

# All hour and year data is the same, therefore it does not return a value
df = df.drop(columns=['hour'])
df = df.drop(columns=['year'])

print(df['country'].nunique())

# All country data is the same, therefore it does not return a value
df = df.drop(columns=['country'])

df[['state', 'code']] = df['statezip'].str.split(' ', expand=True)
unique_states = df['state'].nunique()
print(unique_states)

# If we review the zip codes, they all correspond to Washington, therefore it doesn't provide real value.
df = df.drop(columns=['statezip'])
df = df.drop(columns=['state'])
df = df.drop(columns=['code'])

unique_cities = df['city'].unique()
print(unique_cities)

unique_cities = df['city'].unique()
label_encoder = LabelEncoder()
df['city_map'] = label_encoder.fit_transform(df['city'])
city_mapping = dict(zip(unique_cities, label_encoder.transform(unique_cities)))
print("city mapping:")
print(city_mapping)

df = df.drop(columns=['city'])

# In the case of 'street', there are 4525 different addresses, therefore this variable will not be used.
unique_cities = df['street'].nunique()
print(unique_cities)

df = df.drop(columns=['street'])

 # By convention, outliers correspond to:"
 # outliers <= q1 - 1.5 * irq or  >= q3 + 1.5 * irq
 # where irq = Interquartile range or q3 - q1

def perc_outliers (df):
  for k, v in df.items():
    if v.dtype in ['float64', 'int64']:
      q1 = v.quantile(0.25)
      q3 = v.quantile(0.75)
      irq = q3 - q1
      v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
      perc = np.shape(v_col)[0] * 100.0 / np.shape(df)[0]
      print(f' Column {k} outliers = {perc:.2f}%')

perc_outliers_df = perc_outliers(df)

def del_outliers(df, columns):
  df_out = df.copy()
  for column in columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_out = df_out[(df_out[column] >= lower_bound) & (df_out[column] <= upper_bound)]
  return df_out

columnas = ['sqft_lot', 'price']
df_out = del_outliers(df, columnas)

# Sesgo
def skew (df):
  for k, v in df.items():
    if v.dtype in ['float64', 'int64']:
      df_skew = v.skew()
      print(f' Coolumn {k} skew = {df_skew:.2f}%')

df_skew = skew(df_out)

correlation = df_out.corr()
plt.figure(figsize=(15,15))
ax = sns.heatmap(correlation, vmax = 1, square = True, annot = True, cmap ='viridis')
plt.title("Matriz de correlación")
plt.show()

# The variables that have the strongest correlation with price are: 
# sqft_living (0.56), sqft_above(0.46), bathrooms (0.41), bedrooms (0.27), 
# floors (0.25), sqft_basement(0.23), view (0.21) and city_map (0.12)

# However, there are variables that have a high correlation with each other: 
# sqft_above with sqft_living, sqft_living with bathrooms, and sqft_above with bathrooms. 
# Therefore, we should remove 2 of them.

df_out[["sqft_above","sqft_living","bathrooms","price"]].corr()

new_columns = ['sqft_living', 'sqft_above', 'floors', 'sqft_basement', 'view', 'city_map', 'price']
copy_df = df[new_columns].copy()