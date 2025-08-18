import pandas as pd

url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vS0hLsLJGReYZeR3SI7Z_5R79xK8wK7YfXnC7GaogzYGWyx-lAsJMioL3jyIX5vjPXImPyln9DABrXe/pub?gid=0&single=true&output=csv'

df = pd.read_csv(url, index_col=0)
df['Change'] = df['Close'].pct_change()
print(df.head(10).to_markdown())
