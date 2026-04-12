import pandas as pd

# Load the central dataset
livable_df = pd.read_csv('livable_cities.csv')

# Load cost-of-living dataset and select relevant columns
cost_df = pd.read_csv('cost-of-living.csv', usecols=['city', 'country', 'x1', 'x54'])
cost_df.rename(columns={'city': 'City', 'country': 'Country'}, inplace=True)

# Merge cost-of-living data
livable_df = livable_df.merge(cost_df, on=['City', 'Country'], how='left')

# Load air pollution dataset and select relevant columns
pollution_df = pd.read_csv('air_pollution.csv', usecols=['city', 'country', '2023'])
pollution_df.rename(columns={'city': 'City', 'country': 'Country', '2023': 'Air_Pollution_2023'}, inplace=True)

# Merge air pollution data
livable_df = livable_df.merge(pollution_df, on=['City', 'Country'], how='left')

# Load uaScoresDataFrame dataset and select relevant columns
ua_df = pd.read_csv('uaScoresDataFrame.csv', usecols=['UA_Name', 'UA_Country', 'Education', 'Taxation', 'Internet Access'])
ua_df.rename(columns={'UA_Name': 'City', 'UA_Country': 'Country'}, inplace=True)

# Strip whitespace from City and Country columns
ua_df['City'] = ua_df['City'].str.strip()
ua_df['Country'] = ua_df['Country'].str.strip()

# Normalize country names to match livable_cities.csv
country_mapping = {
    'New Mexico': 'UnitedStates',
    'Alabama': 'UnitedStates',
    'Alaska': 'UnitedStates',
    'Georgia': 'UnitedStates',
    'Texas': 'UnitedStates',
    'Maryland': 'UnitedStates',
    'North Carolina': 'UnitedStates',
    'South Carolina': 'UnitedStates',
    'Colorado': 'UnitedStates',
    'Montana': 'UnitedStates',
    'Idaho': 'UnitedStates',
    'Nevada': 'UnitedStates',
    'California': 'UnitedStates',
    'Washington': 'UnitedStates',
    'Oregon': 'UnitedStates',
    'New York': 'UnitedStates',
    'New Zealand': 'NewZealand',
    'Hong Kong': 'HongKong(China)',
    'South Korea': 'SouthKorea',
    'Saudi Arabia': 'SaudiArabia',
    'United Kingdom': 'UnitedKingdom',
    'Czech Republic': 'CzechRepublic',
    'United Arab Emirates': 'UnitedArabEmirates',
    'Bosnia and Herzegovina': 'BosniaAndHerzegovina',
    'North Macedonia': 'NorthMacedonia',
    'South Africa': 'SouthAfrica',
}

print("Before normalization - sample UA countries:")
print(ua_df['Country'].unique()[:20])

ua_df['Country'] = ua_df['Country'].replace(country_mapping)

print("\nAfter normalization - sample UA countries:")
print(ua_df['Country'].unique()[:20])

print("\nUA DataFrame preview before merge:")
print(ua_df.head(15))
print("\nLivable DataFrame City/Country preview:")
print(livable_df[['City', 'Country']].head(15))

# Find matching cities
print("\nDebug - Looking for matches (first 15):")
for idx, row in ua_df.head(15).iterrows():
    matches = livable_df[(livable_df['City'] == row['City']) & (livable_df['Country'] == row['Country'])]
    print(f"{row['City']}, {row['Country']}: {len(matches)} matches")

# Merge ua data
livable_df = livable_df.merge(ua_df, on=['City', 'Country'], how='left')

# Save the merged dataset
livable_df.to_csv('merged_livable_cities.csv', index=False)

print("\n\nMerged dataset saved as merged_livable_cities.csv")
print(f"\nTotal rows in merged dataset: {len(livable_df)}")
print(f"Rows with Education data: {livable_df['Education'].notna().sum()}")
print(f"Rows with Taxation data: {livable_df['Taxation'].notna().sum()}")
print(f"Rows with Internet Access data: {livable_df['Internet Access'].notna().sum()}")