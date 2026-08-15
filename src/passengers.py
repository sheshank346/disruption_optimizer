"""
passengers.py
Simulates passengers booked on a given flight, with features that a real
airline system would have: fare class, loyalty tier, connection tightness,
booking lead time, historical no-show behavior, etc.

We also generate a LABEL for training: 'rebooking_priority' -- whether this
passenger should be treated as high priority when reaccommodating after a
disruption. In the real world this label would come from historical ops data
(who airlines actually prioritized). Here we simulate it using a realistic
rule-based process with noise, which is standard practice when building a
portfolio ML project without access to proprietary airline data.
"""

import pandas as pd
import numpy as np

FARE_CLASSES = ["Economy", "Premium Economy", "Business", "First"]
FARE_WEIGHTS = [0.65, 0.20, 0.12, 0.03]

LOYALTY_TIERS = ["None", "Silver", "Gold", "Platinum"]
LOYALTY_WEIGHTS = [0.55, 0.25, 0.15, 0.05]


def generate_passengers(flight_id: str, num_passengers: int,
                         connecting_flight_available: bool,
                         min_connection_time: float = None,
                         seed=None) -> pd.DataFrame:
    """
    Generates a synthetic passenger manifest for a disrupted flight.
    """
    rng = np.random.default_rng(seed)

    fare_class = rng.choice(FARE_CLASSES, size=num_passengers, p=FARE_WEIGHTS)
    loyalty_tier = rng.choice(LOYALTY_TIERS, size=num_passengers, p=LOYALTY_WEIGHTS)
    booking_lead_days = rng.exponential(scale=20, size=num_passengers).round().astype(int)
    has_checked_bag = rng.choice([True, False], size=num_passengers, p=[0.7, 0.3])
    group_size = rng.choice([1, 2, 3, 4], size=num_passengers, p=[0.5, 0.3, 0.15, 0.05])
    past_no_show_rate = rng.beta(2, 20, size=num_passengers)  # most passengers rarely no-show
    has_connecting_flight = rng.choice(
        [True, False], size=num_passengers,
        p=[0.55, 0.45] if connecting_flight_available else [0.0, 1.0]
    )
    # connection buffer only meaningful if they have a connecting flight
    connection_buffer_min = np.where(
        has_connecting_flight,
        rng.integers(45, 240, size=num_passengers),
        0
    )

    df = pd.DataFrame({
        "passenger_id": [f"{flight_id}_P{i:04d}" for i in range(num_passengers)],
        "flight_id": flight_id,
        "fare_class": fare_class,
        "loyalty_tier": loyalty_tier,
        "booking_lead_days": booking_lead_days,
        "has_checked_bag": has_checked_bag,
        "group_size": group_size,
        "past_no_show_rate": past_no_show_rate.round(3),
        "has_connecting_flight": has_connecting_flight,
        "connection_buffer_min": connection_buffer_min,
    })

    df["rebooking_priority"] = _simulate_priority_label(df, rng)
    return df


def _simulate_priority_label(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """
    Rule-based simulation of which passengers airlines would realistically
    prioritize when reaccommodating after a disruption:
      - Higher fare class / loyalty tier -> higher priority (contractual + revenue reasons)
      - Tight/at-risk connections -> higher priority (higher risk of full missed journey)
      - Larger groups (esp. families) -> higher priority (harder to split, PR risk)
    Adds noise so the pattern isn't perfectly linear -- makes it a real learning
    problem rather than a lookup table.
    """
    fare_score = df.fare_class.map({"Economy": 0, "Premium Economy": 1, "Business": 2, "First": 3})
    loyalty_score = df.loyalty_tier.map({"None": 0, "Silver": 1, "Gold": 2, "Platinum": 3})

    connection_risk = np.where(
        df.has_connecting_flight,
        np.clip(1 - (df.connection_buffer_min / 240), 0, 1),  # tighter buffer -> higher risk
        0
    )

    group_score = (df.group_size - 1) / 3  # normalize 0-1

    raw_score = (
        0.30 * (fare_score / 3) +
        0.25 * (loyalty_score / 3) +
        0.30 * connection_risk +
        0.15 * group_score
    )

    noise = rng.normal(0, 0.08, size=len(df))
    final_score = np.clip(raw_score + noise, 0, 1)

    # top ~40% become "high priority" (1), rest "standard" (0)
    threshold = np.quantile(final_score, 0.60)
    return (final_score >= threshold).astype(int)


if __name__ == "__main__":
    passengers = generate_passengers(
        flight_id="6E1010",
        num_passengers=180,
        connecting_flight_available=True,
        seed=42
    )
    print(passengers.head(10))
    print(f"\nTotal passengers: {len(passengers)}")
    print(f"High priority: {passengers.rebooking_priority.sum()} "
          f"({passengers.rebooking_priority.mean()*100:.1f}%)")
