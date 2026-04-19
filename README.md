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

## User Input and Recommendation Output

This project now supports a user-driven city recommendation flow.

### How user gives input
- Run `recommend_city.py`.
- The script asks for preference importance scores from `0` to `10` for each key attribute.
- Example meaning: `Safety Index = 10` means safety is very important, `Taxation = 0` means taxation is not important.

### What output is given
- The model trains on `processed_livable_cities.csv` and predicts livability for each city.
- A final suitability score is computed by combining:
  - model livability score, and
  - user preference match score.
- The script prints:
  - the best city for the selected conditions,
  - top 5 suitable cities.
- It also saves the full ranked list to `city_recommendation_results.csv`.