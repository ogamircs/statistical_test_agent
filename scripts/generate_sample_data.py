"""
Generate Sample A/B Test Data

Creates a realistic sample CSV dataset for testing the A/B testing agent.
Includes multiple customer segments with varying treatment effects.

Usage:
    python scripts/generate_sample_data.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def generate_sample_data(
    n_customers: int = 5000,
    output_path: str = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate sample A/B test data with realistic characteristics

    Parameters:
    - n_customers: Total number of customers
    - output_path: Path to save the CSV file (defaults to data/sample_ab_data.csv)
    - seed: Random seed for reproducibility
    """
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "sample_ab_data.csv")

    np.random.seed(seed)

    # Define segments with different characteristics
    segments = {
        "Premium": {
            "proportion": 0.15,
            "base_effect": 50.0,
            "treatment_lift": 8.0,
            "effect_std": 15.0
        },
        "Standard": {
            "proportion": 0.45,
            "base_effect": 30.0,
            "treatment_lift": 3.0,
            "effect_std": 12.0
        },
        "Basic": {
            "proportion": 0.25,
            "base_effect": 15.0,
            "treatment_lift": 0.5,
            "effect_std": 8.0
        },
        "New": {
            "proportion": 0.15,
            "base_effect": 20.0,
            "treatment_lift": -2.0,
            "effect_std": 10.0
        }
    }

    # Generate customer data
    data = []
    customer_id = 1

    for segment_name, params in segments.items():
        n_segment = int(n_customers * params["proportion"])

        for _ in range(n_segment):
            is_treatment = np.random.random() < 0.6
            group = "treatment" if is_treatment else "control"

            base_value = np.random.normal(
                params["base_effect"],
                params["effect_std"]
            )

            if is_treatment:
                effect_value = base_value + params["treatment_lift"] + np.random.normal(0, 2)
            else:
                effect_value = base_value

            effect_value = max(0, effect_value)
            duration_days = np.random.randint(7, 31)

            start_date = datetime(2024, 1, 1)
            signup_days = np.random.randint(0, 365)
            signup_date = start_date + timedelta(days=signup_days)

            data.append({
                "customer_id": f"CUST_{customer_id:05d}",
                "experiment_group": group,
                "customer_segment": segment_name,
                "effect_value": round(effect_value, 2),
                "experiment_duration_days": duration_days,
                "signup_date": signup_date.strftime("%Y-%m-%d"),
                "region": np.random.choice(["North", "South", "East", "West"]),
                "age_group": np.random.choice(["18-25", "26-35", "36-45", "46-55", "55+"]),
                "is_mobile": np.random.choice([True, False])
            })

            customer_id += 1

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} customer records")
    print(f"Saved to: {output_path}")
    print(f"\nSegment distribution:")
    print(df["customer_segment"].value_counts())
    print(f"\nGroup distribution:")
    print(df["experiment_group"].value_counts())

    return df


def generate_alternative_format(
    n_customers: int = 3000,
    output_path: str = None,
    seed: int = 123
) -> pd.DataFrame:
    """
    Generate data with different column naming conventions
    to test the agent's column detection flexibility
    """
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "sample_ab_data_alt.csv")

    np.random.seed(seed)

    data = []

    tiers = ["Gold", "Silver", "Bronze"]
    tier_effects = {
        "Gold": (100, 20, 15),
        "Silver": (60, 8, 12),
        "Bronze": (30, 2, 8)
    }

    for i in range(n_customers):
        tier = np.random.choice(tiers, p=[0.2, 0.4, 0.4])
        base, lift, std = tier_effects[tier]

        is_test = np.random.random() < 0.5
        variant = "test" if is_test else "ctrl"

        value = np.random.normal(base, std)
        if is_test:
            value += lift + np.random.normal(0, 3)

        data.append({
            "user_id": i + 1,
            "ab_variant": variant,
            "tier": tier,
            "revenue": round(max(0, value), 2),
            "days_in_experiment": np.random.randint(14, 45)
        })

    df = pd.DataFrame(data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nAlternative format generated: {output_path}")
    print(f"Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    print("Generating sample A/B test data...")
    print("=" * 50)

    df1 = generate_sample_data()

    print("\n" + "=" * 50 + "\n")

    df2 = generate_alternative_format()

    print("\n" + "=" * 50)
    print("\nSample data files generated successfully!")
    print(f"Files saved to: {DATA_DIR}")
    print("Use these files to test the A/B Testing Agent.")
