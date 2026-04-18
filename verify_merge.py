import pandas as pd

MEAL_INEXPENSIVE_USD = "Meal, Inexpensive Restaurant (USD)"
AVG_MONTHLY_NET_SALARY_AFTER_TAX = "Average Monthly Net Salary (After Tax)"

df = pd.read_csv('merged_livable_cities.csv')
print('Merged Dataset Info:')
print(f'Total rows: {len(df)}')
print(f'Total columns: {len(df.columns)}')
print(f'\nColumns: {df.columns.tolist()}')
print(f'\nRows with Education data: {df["Education"].notna().sum()}')
print(f'Rows with Taxation data: {df["Taxation"].notna().sum()}')
print(f'Rows with Internet Access data: {df["Internet Access"].notna().sum()}')
print(f'Rows with Air_Pollution_2023 data: {df["Air_Pollution_2023"].notna().sum()}')
print(
    f'Rows with {MEAL_INEXPENSIVE_USD} (cost-of-living) data: '
    f'{df[MEAL_INEXPENSIVE_USD].notna().sum()}'
)
print(
    f'Rows with {AVG_MONTHLY_NET_SALARY_AFTER_TAX} (cost-of-living) data: '
    f'{df[AVG_MONTHLY_NET_SALARY_AFTER_TAX].notna().sum()}'
)

print('\nSample row with all merged data:')
sample = df[
    (df["Education"].notna())
    & (df["Air_Pollution_2023"].notna())
    & (df[MEAL_INEXPENSIVE_USD].notna())
].iloc[0]
print(f'City: {sample["City"]}, Country: {sample["Country"]}')
print(f'Education: {sample["Education"]}, Taxation: {sample["Taxation"]}, Internet Access: {sample["Internet Access"]}')
print(
    f'{MEAL_INEXPENSIVE_USD}: {sample[MEAL_INEXPENSIVE_USD]}, '
    f'{AVG_MONTHLY_NET_SALARY_AFTER_TAX}: {sample[AVG_MONTHLY_NET_SALARY_AFTER_TAX]}'
)
print(f'Air Pollution 2023: {sample["Air_Pollution_2023"]}')
