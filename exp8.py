import pandas as pd 
from sklearn.preprocessing import OneHotEncoder 
data = {'Employee id': [10, 20, 15, 25, 30], 
        'Gender': ['M', 'F', 'F', 'M', 'F'], 
        'Remarks': ['Good', 'Nice', 'Good', 'Great', 'Nice'], 
        } 
df = pd.DataFrame(data) 
print(f"Employee data : \n{df}") 
#Here we extract the columns with object datatype as they are the categorical columns 
categorical_columns = df.select_dtypes(include=['object']).columns.tolist() 
encoder = OneHotEncoder(sparse_output=False) 
# Apply one-hot encoding to the categorical columns 
one_hot_encoded = encoder.fit_transform(df[categorical_columns]) 
one_hot_df =pd.DataFrame(one_hot_encoded, 
columns=encoder.get_feature_names_out(categorical_columns)) 
df_encoded = pd.concat([df, one_hot_df], axis=1) 
df_encoded = df_encoded.drop(categorical_columns, axis=1) 
print(f"Encoded Employee data : \n{df_encoded}") 