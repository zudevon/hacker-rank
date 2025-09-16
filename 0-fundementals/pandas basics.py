# Quick setup
import pandas as pd
import numpy as np

# Create Series / DataFrame
s = pd.Series([10, 20, 30], index=['a','b','c'])
df = pd.DataFrame({
    'id': [1,2,3],
    'name': ['alice','bob','carol'],
    'age': [25, 30, 22],
    'score': [85.5, 92.0, 78.0]
})
# From numpy
arr = np.random.rand(4,3)
df2 = pd.DataFrame(arr, columns=['A','B','C'])

# Quick inspection
df.head()        # first 5 rows
df.tail(3)       # last 3 rows
df.shape         # (rows, cols)
df.info()        # dtypes, non-null counts
df.describe()    # numeric summary (count, mean, std, min, 25/50/75%, max)
df.dtypes

# Indexing: label vs positional
df.loc[0]        # row label 0 (label-based)
df.loc[0, 'name']
df.loc[:, ['name','score']]   # label-based column selection

df.iloc[0]       # first row (positional)
df.iloc[:, 2]    # 3rd column (positional)

# Boolean selection / masks
df[df['age'] > 24]
mask = (df.age > 24) & (df.score > 80)
df[mask]

# .at, .iat, .loc best practices
#
# Use .loc for label-based selection, .iloc for positional.
#
# Use .at and .iat for fast scalar access/assignment:

df.at[1, 'name'] = 'robert'
df.iat[2, 3] = 80.0

# Add / modify / drop columns
df['passed'] = df['score'] >= 80
df['age_plus_one'] = df['age'] + 1
df.drop(columns=['age_plus_one'], inplace=True)

# Vectorized operations (avoid Python loops)
df['zscore'] = (df['score'] - df['score'].mean()) / df['score'].std()

# Missing data
df2 = pd.DataFrame({'a':[1, np.nan, 3], 'b':[np.nan, 2, 3]})
df2.isna()             # boolean mask
df2.dropna()           # drop rows with any NaN
df2.fillna(0)          # replace NaN with value
df2.fillna(method='ffill')  # forward-fill

# GroupBy & aggregation
# single aggregation
df.groupby('passed')['score'].mean()

# multi-agg
df.groupby('passed').agg(
    count=('id','size'),
    avg_score=('score','mean'),
    max_score=('score','max')
)

# group and transform (preserve shape)
df['score_z_by_pass'] = df.groupby('passed')['score'].transform(lambda x: (x - x.mean())/x.std())

# Sorting
df.sort_values('score', ascending=False)
df.sort_values(['passed','score'], ascending=[True, False])
df.sort_index()   # sort by index

# Merge / Join (SQL-like)
left = pd.DataFrame({'id':[1,2,3], 'city':['NY','LA','SF']})
right = pd.DataFrame({'id':[2,3,4], 'value':[10,20,30]})

pd.merge(left, right, on='id', how='inner')   # inner, left, right, outer

# Concatenate / append
pd.concat([df1, df2], axis=0)   # stack rows (use ignore_index=True if needed)
pd.concat([dfA, dfB], axis=1)   # join columns

# Reshape: pivot, melt, pivot_table
# wide -> long
df_long = pd.melt(df, id_vars=['id','name'], value_vars=['age','score'])

# pivot (long -> wide)
df_pivot = df_long.pivot(index='id', columns='variable', values='value')

# pivot_table with aggregation
pd.pivot_table(df, index='passed', values='score', aggfunc=['mean','count'])

# Time series & datetime
df['date'] = pd.to_datetime(['2022-01-01','2022-01-02','2022-01-05'])
df.set_index('date', inplace=True)
df.resample('D').mean()         # daily resample (fill missing days)
df.index.to_period('M')         # convert to period
df['month'] = df.index.month

# Window functions
df['rolling_mean_3'] = df['score'].rolling(window=3, min_periods=1).mean()
df['expma'] = df['score'].ewm(span=5).mean()

# Apply vs vectorized methods
#
# Prefer built-in vectorized ufuncs and pandas methods.
#
# .apply() is flexible but slower (calls Python function per row/column).

df['initial'] = df['name'].str[0]    # vectorized string op
df['name_len'] = df['name'].apply(len)  # slower than vectorized alternatives

# String operations
df['name_upper'] = df['name'].str.upper()
df['contains_o'] = df['name'].str.contains('o')

# I/O: read / write common formats
pd.read_csv('data.csv', parse_dates=['timestamp'])
df.to_csv('out.csv', index=False)

# faster / binary formats
df.to_parquet('out.parquet')   # preferred for large datasets
pd.read_parquet('out.parquet')
df.to_pickle('out.pkl')