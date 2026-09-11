"""
generate_dataset.py
--------------------
Generates a realistic SYNTHETIC house price dataset for the
House Price Prediction ML project.

Why synthetic data?
Real-world housing datasets are either paid, region-locked, or too
large/messy for a beginner-friendly portfolio project. This script
creates a dataset with realistic relationships (bigger area -> higher
price, more bedrooms/bathrooms -> higher price, older house -> lower
price, location -> price multiplier) plus random noise, so the data
behaves like real data without needing an external download.

Run:
    python generate_dataset.py

Output:
    data/house_prices.csv
"""

import os
import numpy as np
import pandas as pd

# Fix the random seed so the dataset is reproducible every time
# this script is run (important for consistent training results).
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Number of rows to generate. 3000 rows is enough to train a stable
# regression model while staying light enough for Streamlit Cloud.
NUM_SAMPLES = 3000

# Locations and a rough "desirability multiplier" for each one.
# This simulates how the same house can cost more or less purely
# because of where it is located.
LOCATIONS = {
    "Downtown": 1.35,
    "Suburb": 1.05,
    "Rural": 0.80,
    "City Center": 1.50,
    "Industrial Area": 0.90,
}


def generate_dataset(num_samples: int = NUM_SAMPLES) -> pd.DataFrame:
    """Generate a synthetic house price dataset as a pandas DataFrame."""

    # --- Feature generation ---

    # Area in square feet: houses generally range from ~500 to ~4500 sqft
    area_sqft = np.random.normal(loc=1800, scale=700, size=num_samples)
    area_sqft = np.clip(area_sqft, 400, 5000).round(0)

    # Bedrooms: most houses have 1-5 bedrooms
    bedrooms = np.random.choice(
        [1, 2, 3, 4, 5], size=num_samples, p=[0.10, 0.30, 0.35, 0.18, 0.07]
    )

    # Bathrooms: usually correlated with bedrooms, but not always equal
    bathrooms = np.clip(
        bedrooms - np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4]),
        1,
        5,
    )

    # Location: random categorical feature
    location = np.random.choice(list(LOCATIONS.keys()), size=num_samples)

    # Parking: 0 = No, 1 = Yes
    parking = np.random.choice([0, 1], size=num_samples, p=[0.35, 0.65])

    # House age in years: 0 (new) to 40 years old
    house_age = np.random.randint(0, 41, size=num_samples)

    # --- Target generation (price in Lakhs INR) ---

    # Base price driven mainly by area (price per sqft in "Lakhs" units)
    base_price_per_sqft = 0.032  # ~ 32,000 INR per sqft baseline

    price = area_sqft * base_price_per_sqft

    # Extra value added per bedroom and bathroom
    price += bedrooms * 4.5
    price += bathrooms * 3.0

    # Parking adds a fixed premium
    price += parking * 5.0

    # Older houses are worth less (depreciation), but the effect
    # flattens out for very old houses (floor at ~40% of value lost)
    depreciation_factor = np.clip(1 - (house_age * 0.012), 0.55, 1.0)
    price = price * depreciation_factor

    # Apply location multiplier
    location_multiplier = np.array([LOCATIONS[loc] for loc in location])
    price = price * location_multiplier

    # Add random noise so the relationship isn't perfectly clean
    # (this mimics real-world unpredictability in housing prices)
    noise = np.random.normal(loc=0, scale=6, size=num_samples)
    price = price + noise

    # Prices should never be negative or unrealistically small
    price = np.clip(price, 8, None).round(2)

    # --- Assemble the DataFrame ---
    df = pd.DataFrame(
        {
            "area_sqft": area_sqft.astype(int),
            "bedrooms": bedrooms.astype(int),
            "bathrooms": bathrooms.astype(int),
            "location": location,
            "parking": parking.astype(int),
            "house_age": house_age.astype(int),
            "price": price,  # target column, in Lakhs (INR)
        }
    )

    return df


def main():
    df = generate_dataset()

    # Make sure the "data" folder exists relative to this script,
    # so the project works no matter which directory it's run from.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, "house_prices.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Synthetic dataset generated successfully!")
    print(f"   Rows: {len(df)}")
    print(f"   Saved to: {output_path}")
    print("\nSample rows:")
    print(df.head())


if __name__ == "__main__":
    main()
