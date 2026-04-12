import pandas as pd
df = pd.read_csv('merged_livable_cities.csv')
print('Merged Dataset Info:')
print(f'Total rows: {len(df)}')
print(f'Total columns: {len(df.columns)}')
print(f'\nColumns: {df.columns.tolist()}')
print(f'\nRows with Education data: {df["Education"].notna().sum()}')
print(f'Rows with Taxation data: {df["Taxation"].notna().sum()}')
print(f'Rows with Internet Access data: {df["Internet Access"].notna().sum()}')
print(f'Rows with Air_Pollution_2023 data: {df["Air_Pollution_2023"].notna().sum()}')
print(f'Rows with x1 (cost-of-living) data: {df["x1"].notna().sum()}')
print(f'Rows with x54 (cost-of-living) data: {df["x54"].notna().sum()}')

print('\nSample row with all merged data:')
sample = df[(df["Education"].notna()) & (df["Air_Pollution_2023"].notna()) & (df["x1"].notna())].iloc[0]
print(f'City: {sample["City"]}, Country: {sample["Country"]}')
print(f'Education: {sample["Education"]}, Taxation: {sample["Taxation"]}, Internet Access: {sample["Internet Access"]}')
print(f'x1: {sample["x1"]}, x54: {sample["x54"]}')
print(f'Air Pollution 2023: {sample["Air_Pollution_2023"]}')
