import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load your dataset
data = pd.read_csv('carz_cleaned.csv')

# Select features and target
features = ['make', 'model', 'year']
target = 'sellingprice'

# Prepare the input and target data
X = data[features]
y = data[target]

# Preprocess the features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['year']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['make', 'model'])
    ])

# Fit and transform the features
X_processed = preprocessor.fit_transform(X)

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_processed, y)

# Save the trained model
joblib.dump(rf_model, 'car_price_prediction_model.joblib')

# Now let's define a function to predict the selling price
def predict_selling_price(make, model, year):
    # Transform the input features
    input_data = pd.DataFrame({'make': [make], 'model': [model], 'year': [year]})
    input_processed = preprocessor.transform(input_data)
    loaded_model = joblib.load('car_price_prediction_model.joblib')
    predicted_price = loaded_model.predict(input_processed)
    return predicted_price[0]

