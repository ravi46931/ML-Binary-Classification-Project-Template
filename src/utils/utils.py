
import pandas as pd
import numpy as np

def modify_null_values(df, column):
    null_count=df[column].isna().sum()
    total_count=df[column].value_counts().sum()  # without null
    for i in range(len(df[column].value_counts())):
        null_indices = df[df[column].isna()].index
        count_cat= df[column].value_counts()[i]
        val=(null_count*count_cat)/total_count
        val=round(val)
        indices_to_update = np.random.choice(null_indices, size=min(val, len(null_indices)), replace=False)
        specific_category = df[column].value_counts().index[i]
        df.loc[indices_to_update, column] = specific_category

    return df

def frequency_encoding(df, column):
    dictionary={}
 
    for i in range(len(df[column].value_counts())):
        value = (df[column].value_counts()[i]) / (df[column].value_counts().sum())
        value=round(value, 4)
        key=df[column].value_counts().index[i]
        dictionary[key]=value
      
    return dictionary

def label_encoding(df, column):
    dictionary={}
    for i in range(len(df[column].value_counts())):
        key=df[column].value_counts().index[i]
        dictionary[key]=i
      
    return dictionary

# Split the data into train and test part
def data_split(df,test_size_percentage=0.25):
    test_size=int(df.shape[0]*test_size_percentage)
    test_data = df.sample(n=test_size, random_state=42)
    train_data = df[~df.isin(test_data)].dropna()
    return train_data, test_data

# Seperate the dependent and independent variables
def get_features_target(data, dependent_variable):
    X=data.drop(dependent_variable, axis=1)
    y=data[dependent_variable]
    return X, y
