# City-Prediction-Project-ML

## Project Overview
This project aims to create a model that predicts the best city to live in based on various livability factors.

## Phases

### Phase 1: Data Merging
- **Objective**: Merge relevant features from auxiliary datasets into the central livable_cities.csv dataset.
- **Datasets Used**:
  - livable_cities.csv (central)
  - cost-of-living.csv (merged Meal, Inexpensive Restaurant (USD) and Average Monthly Net Salary (After Tax) columns)
  - air_pollution.csv (merged 2023 column)
  - uaScoresDataFrame.csv (merged Education, Taxation, Internet Access columns)
- **Output**: merged_livable_cities.csv
- **Notes**: See data_collection.txt for detailed preprocessing tasks.