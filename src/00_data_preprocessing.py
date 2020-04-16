from dsproject import dsproject

dsp = dsproject()
dsp.gen_directories()

from preprocess import *

# Clean DBSOC data
df = dsp.read_data('DB_SOC')
df = transform_range_day(df)
df = transform_group(df)
df = delete_missing_columns(df)
df = replace_nan(df)
df = df.drop_duplicates()

print("Min-Max FileDate")
print(df['FileDate'].min())
print(df['FileDate'].max())
print(df.shape)

dsp.write_table(df, 'DB_SOC', 'preprocess')
