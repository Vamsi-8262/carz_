import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_and_preprocess_data():
    data = pd.read_csv('carz_cleaned.csv')
    data.dropna(subset=['sellingprice'], inplace=True)
    features = ['make', 'model', 'year']
    target = 'sellingprice'
    X = data[features]
    y = data[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['year']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['make', 'model'])
        ])
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y

def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict(model, X):
    return model.predict(X)

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2
