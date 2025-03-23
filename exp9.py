import pandas as pd 
from sklearn.preprocessing import LabelEncoder 
data = { 
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Green'], 
    'Size': ['Small', 'Large', 'Medium', 'Medium', 'Small'], 
    'Shape': ['Circle', 'Square', 'Triangle', 'Circle', 'Square'], 
    'Label': ['A', 'B', 'C', 'A', 'B'] 
} 
df = pd.DataFrame(data) 
print("Original dataset:") 
print(df) 
label_encoder = LabelEncoder() 
df_encoded = df.copy() 
for col in df.columns: 
    if df[col].dtype == 'object': 
        df_encoded[col] = label_encoder.fit_transform(df[col]) 
print("\nAfter Label Encoding:") 
print(df_encoded)