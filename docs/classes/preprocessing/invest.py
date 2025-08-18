import pandas as pd

url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vS0hLsLJGReYZeR3SI7Z_5R79xK8wK7YfXnC7GaogzYGWyx-lAsJMioL3jyIX5vjPXImPyln9DABrXe/pub?gid=0&single=true&output=csv'

df = pd.read_csv(url, index_col=0)
df['Change'] = df['Close'].pct_change()
df['Z-Volume'] = df['Volume'].apply(lambda x: (x-df['Volume'].mean())/df['Volume'].std())
df['N-Volume'] = df['Volume'].apply(lambda x: (x-df['Volume'].min())/(df['Volume'].max()-df['Volume'].min()))
df['Z-Change'] = df['Change'].apply(lambda x: (x-df['Change'].mean())/df['Change'].std())
df['N-Change'] = df['Change'].apply(lambda x: (x-df['Change'].min())/(df['Change'].max()-df['Change'].min()))
df = df[['Volume', 'N-Volume', 'Z-Volume', 'Change', 'N-Change', 'Z-Change']].dropna()
print(df.head(10).to_markdown())
