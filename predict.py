import joblib
import pandas as pd

from preprocessing import preprocess_data

# Load trained model
model = joblib.load("model.pkl")

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load encoders
encoders = joblib.load("encoders.pkl")

# Load feature names
feature_names = joblib.load("feature_names.pkl")


def encode_input(df):

    for column, encoder in encoders.items():

        if column in df.columns:

            value = str(df.loc[0, column])

            if value not in encoder.classes_:
                value = encoder.classes_[0]

            df[column] = encoder.transform([value])

    return df

def predict_eta(df):

    df = encode_input(df)

    df = df[feature_names]
    # Scale only if model is Linear Regression
    if model.__class__.__name__ == "LinearRegression":
        df = scaler.transform(df)

    prediction = model.predict(df)

    return round(float(prediction[0]), 2)
