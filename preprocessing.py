import pandas as pd
import numpy as np
from geopy.distance import geodesic


# ---------------------------------------------------------
# Extract useful information from existing columns
# ---------------------------------------------------------
def extract_column_value(df):

    # Only during training
    if "Time_taken(min)" in df.columns:
        df["Time_taken(min)"] = (
            df["Time_taken(min)"]
            .apply(lambda x: int(str(x).split(" ")[1].strip()))
        )

    # Weather
    if "Weatherconditions" in df.columns:
        df["Weatherconditions"] = (
            df["Weatherconditions"]
            .astype(str)
            .apply(lambda x: x.split(" ")[1] if " " in x else x)
        )

    # City code
    if "Delivery_person_ID" in df.columns:
        df["City_code"] = (
            df["Delivery_person_ID"]
            .astype(str)
            .str.split("RES", expand=True)[0]
        )

    return df


# ---------------------------------------------------------
# Drop unnecessary columns
# ---------------------------------------------------------
def drop_columns(df):

    columns_to_drop = [
        "ID",
        "Delivery_person_ID"
    ]

    df.drop(
        columns=columns_to_drop,
        errors="ignore",
        inplace=True
    )

    return df


# ---------------------------------------------------------
# Update data types
# ---------------------------------------------------------
def update_datatype(df):
    """
    Convert columns into appropriate data types.
    """

    df["Delivery_person_Age"] = df["Delivery_person_Age"].astype(float)

    df["Delivery_person_Ratings"] = df[
        "Delivery_person_Ratings"
    ].astype(float)

    df["multiple_deliveries"] = df[
        "multiple_deliveries"
    ].astype(float)

    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"],
        format="%d-%m-%Y"
    )

    return df


# ---------------------------------------------------------
# Replace string 'NaN' with actual np.nan
# ---------------------------------------------------------
def convert_nan(df):
    """
    Replace string 'NaN' values with actual missing values.
    """

    df.replace("NaN", np.nan, inplace=True)

    return df


# ---------------------------------------------------------
# Handle missing values
# ---------------------------------------------------------
def handle_null_values(df):
    """
    Fill missing values.
    """

    df["Delivery_person_Age"] = df[
        "Delivery_person_Age"
    ].fillna(
        np.random.choice(df["Delivery_person_Age"].dropna())
    )

    df["Weatherconditions"] = df[
        "Weatherconditions"
    ].fillna(
        np.random.choice(df["Weatherconditions"].dropna())
    )

    df["City"] = df["City"].fillna(
        df["City"].mode()[0]
    )

    df["Festival"] = df["Festival"].fillna(
        df["Festival"].mode()[0]
    )

    df["multiple_deliveries"] = df[
        "multiple_deliveries"
    ].fillna(
        df["multiple_deliveries"].mode()[0]
    )

    df["Road_traffic_density"] = df[
        "Road_traffic_density"
    ].fillna(
        df["Road_traffic_density"].mode()[0]
    )

    df["Delivery_person_Ratings"] = df[
        "Delivery_person_Ratings"
    ].fillna(
        df["Delivery_person_Ratings"].median()
    )

    return df


# ---------------------------------------------------------
# Date Feature Engineering
# ---------------------------------------------------------
def extract_date_features(df):
    """
    Create useful date features.
    """

    df["day"] = df["Order_Date"].dt.day

    df["month"] = df["Order_Date"].dt.month

    df["quarter"] = df["Order_Date"].dt.quarter

    df["year"] = df["Order_Date"].dt.year

    df["day_of_week"] = df[
        "Order_Date"
    ].dt.day_of_week.astype(int)

    df["is_month_start"] = df[
        "Order_Date"
    ].dt.is_month_start.astype(int)

    df["is_month_end"] = df[
        "Order_Date"
    ].dt.is_month_end.astype(int)

    df["is_quarter_start"] = df[
        "Order_Date"
    ].dt.is_quarter_start.astype(int)

    df["is_quarter_end"] = df[
        "Order_Date"
    ].dt.is_quarter_end.astype(int)

    df["is_year_start"] = df[
        "Order_Date"
    ].dt.is_year_start.astype(int)

    df["is_year_end"] = df[
        "Order_Date"
    ].dt.is_year_end.astype(int)

    df["is_weekend"] = np.where(
        df["day_of_week"].isin([5, 6]),
        1,
        0,
    )

    return df


# ---------------------------------------------------------
# Order preparation time
# ---------------------------------------------------------
def calculate_time_diff(df):


    df["Time_Orderd"] = (
        df["Time_Orderd"]
        .astype(str)
        .str.strip()
        .replace("NaN", np.nan)
    )

    df["Time_Order_picked"] = (
        df["Time_Order_picked"]
        .astype(str)
        .str.strip()
        .replace("NaN", np.nan)
    )
    """
    Calculate order preparation time.
    """

    df["Time_Orderd"] = pd.to_timedelta(df["Time_Orderd"])

    df["Time_Order_picked"] = pd.to_timedelta(
        df["Time_Order_picked"]
    )

    df["Time_Order_picked_formatted"] = (
        df["Order_Date"] + df["Time_Order_picked"]
    )

    mask = df["Time_Order_picked"] < df["Time_Orderd"]

    df.loc[
        mask,
        "Time_Order_picked_formatted"
    ] += pd.DateOffset(days=1)

    df["Time_Ordered_formatted"] = (
        df["Order_Date"] + df["Time_Orderd"]
    )

    df["order_prepare_time"] = (
        df["Time_Order_picked_formatted"]
        - df["Time_Ordered_formatted"]
    ).dt.total_seconds() / 60

    df["order_prepare_time"] = df[
        "order_prepare_time"
    ].fillna(
        df["order_prepare_time"].median()
    )

    df.drop(
        columns=[
            "Time_Orderd",
            "Time_Order_picked",
            "Time_Ordered_formatted",
            "Time_Order_picked_formatted",
            "Order_Date",
        ],
        inplace=True,
    )

    return df


# ---------------------------------------------------------
# Distance Calculation
# ---------------------------------------------------------
def calculate_distance(df):
    """
    Calculate restaurant-to-customer distance.
    """

    restaurant = df[
        [
            "Restaurant_latitude",
            "Restaurant_longitude",
        ]
    ].to_numpy()

    delivery = df[
        [
            "Delivery_location_latitude",
            "Delivery_location_longitude",
        ]
    ].to_numpy()

    df["distance"] = [
        geodesic(r, d).kilometers
        for r, d in zip(restaurant, delivery)
    ]

    df["distance"] = (
        df["distance"]
        .round(2)
    )

    df.drop(
        columns=[
            "Restaurant_latitude",
            "Restaurant_longitude",
            "Delivery_location_latitude",
            "Delivery_location_longitude",
        ],
        inplace=True,
    )

    return df


# ---------------------------------------------------------
# Complete preprocessing pipeline
# ---------------------------------------------------------
def preprocess_data(df, training=True):

    df = extract_column_value(df)

    df = drop_columns(df)

    df = update_datatype(df)

    df = convert_nan(df)

    df = handle_null_values(df)

    df = extract_date_features(df)

    df = calculate_time_diff(df)

    df = calculate_distance(df)

    return df