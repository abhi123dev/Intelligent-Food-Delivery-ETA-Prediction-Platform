import pandas as pd
import joblib

from preprocessing import preprocess_data

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


df = pd.read_csv("dataset.csv")

print("Original Shape :", df.shape)

df = preprocess_data(df)

print("Processed Shape :", df.shape)


label_encoders = {}

categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(
        df[column].astype(str)
    )

    label_encoders[column] = encoder


TARGET = "Time_taken(min)"

X = df.drop(columns=[TARGET])

y = df[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        random_state=42,
    ),
}

results = {}

best_model = None
best_score = float("-inf")

for name, model in models.items():

    if name == "Linear Regression":

        model.fit(
            X_train_scaled,
            y_train,
        )

        predictions = model.predict(
            X_test_scaled
        )

    else:

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    results[name] = {

        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    if r2 > best_score:

        best_score = r2
        best_model = model

results_df = pd.DataFrame(results).T

print(results_df.sort_values(
    by="R2",
    ascending=False
))

# Save model comparison
results_df.to_csv("model_results.csv", index=True)

# Save feature names
joblib.dump(list(X.columns), "feature_names.pkl")

# Save model
joblib.dump(best_model, "model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Save encoders
joblib.dump(label_encoders, "encoders.pkl")

print("\nModel Saved Successfully!")