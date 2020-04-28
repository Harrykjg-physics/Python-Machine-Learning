import pandas as pd
from io import StringIO
import numpy as np
from sklearn.impute import SimpleImputer

csv_data = ''' 
    A,B,C,D
   1.0,2.0,3.0,4.0
   5.0,6.0,,8.0
   0.0,11.0,12.0,
'''
df = pd.read_csv(StringIO(csv_data))
print(df)
print(df.values)
print(np.nan)

imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)

