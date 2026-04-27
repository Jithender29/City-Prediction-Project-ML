from flask import Flask, request, jsonify, render_template
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path to import modules
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from phase3.recommend_city import (
    train_model, 
    add_engineered_features, 
    build_user_profile_row, 
    score_all_cities, 
    find_closest_cities_by_score,
    PROCESSED_DIR
)

app = Flask(__name__)

# Initialize model and data at startup
print("Loading data and training model...")
df = pd.read_csv(PROCESSED_DIR / "processed_livable_cities.csv")
model, feature_cols = train_model(df)
model_df = add_engineered_features(df)
scored_cities = score_all_cities(df, model, feature_cols)

attributes = {
    "Purchasing Power Index": "higher",
    "Safety Index": "higher",
    "Health Care Index": "higher",
    "Cost of Living Index": "lower",
    "Property Price to Income Ratio": "lower",
    "Traffic Commute Time Index": "lower",
    "Pollution Index": "lower",
    "Climate Index": "higher",
    "Education": "higher",
    "Taxation": "lower",
    "Internet Access": "higher",
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Parse inputs, defaulting to 5 if something is missing
        user_values = {}
        for attr in attributes.keys():
            val = data.get(attr, 5.0)
            user_values[attr] = float(val)
            
        user_profile = build_user_profile_row(
            model_df=model_df,
            feature_cols=feature_cols,
            attribute_directions=attributes,
            user_values=user_values,
        )
        
        target_livability_score = float(model.predict(user_profile[feature_cols])[0])
        
        nearest_cities = find_closest_cities_by_score(
            scored_df=scored_cities,
            target_score=target_livability_score,
            top_n=10
        )
        
        results = []
        for _, row in nearest_cities.iterrows():
            results.append({
                "city": row["City"],
                "country": row["Country"],
                "score": f"{row['Predicted_Livability']:.2f}",
                "distance": f"{row['Score_Distance']:.2f}"
            })
            
        return jsonify({
            "success": True,
            "target_score": f"{target_livability_score:.2f}",
            "recommendations": results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
