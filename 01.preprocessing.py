import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()


df = pd.read_csv('datasets/03-02-2018.csv')

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace = True)

df['Timestamp'] = label_encoder.fit_transform(df['Timestamp'])
X = df.drop('Label', axis=1)
y = df['Label']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

balanced_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='Label')], axis=1)

#Encoding our target variable
balanced_df["Label"].replace('Benign',0,inplace=True)
balanced_df["Label"].replace('Bot',1,inplace=True)

balanced_df.to_csv("datasets/balanced.csv")