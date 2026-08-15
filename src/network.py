"""
network.py
Builds a small flight network (airports + routes) using networkx,
and generates flight instances (specific flight on a specific day/time)
that passengers can be booked on.
"""

import networkx as nx
import pandas as pd
from datetime import datetime, timedelta
import random

# A small, realistic set of Indian + a few international airports
AIRPORTS = ["BLR", "DEL", "BOM", "MAA", "HYD", "CCU", "GOI", "DXB", "SIN"]

# Routes (edges) with typical flight duration in minutes
ROUTES = [
    ("BLR", "DEL", 165), ("DEL", "BLR", 165),
    ("BLR", "BOM", 90),  ("BOM", "BLR", 90),
    ("BLR", "MAA", 60),  ("MAA", "BLR", 60),
    ("DEL", "BOM", 120), ("BOM", "DEL", 120),
    ("DEL", "HYD", 130), ("HYD", "DEL", 130),
    ("BOM", "GOI", 70),  ("GOI", "BOM", 70),
    ("DEL", "CCU", 140), ("CCU", "DEL", 140),
    ("BLR", "DXB", 240), ("DXB", "BLR", 240),
    ("BOM", "DXB", 190), ("DXB", "BOM", 190),
    ("DEL", "SIN", 320), ("SIN", "DEL", 320),
    ("BLR", "SIN", 280), ("SIN", "BLR", 280),
    ("HYD", "BOM", 85),  ("BOM", "HYD", 85),
]

AIRLINES = ["IX", "6E", "AI", "UK"]  # fictional-ish carrier codes for the sim


def build_airport_graph() -> nx.DiGraph:
    """Builds a directed graph of airports connected by routes."""
    G = nx.DiGraph()
    G.add_nodes_from(AIRPORTS)
    for origin, dest, duration in ROUTES:
        G.add_edge(origin, dest, duration_min=duration)
    return G


def generate_flight_schedule(num_days=1, seed=42) -> pd.DataFrame:
    """
    Generates a set of concrete flight instances (flight_id, origin, dest,
    departure time, arrival time, capacity) across the route network.
    This is the 'today's schedule' our disruption will hit.
    """
    random.seed(seed)
    flights = []
    flight_counter = 1000
    base_date = datetime(2026, 8, 15, 6, 0)  # 6 AM start

    for day in range(num_days):
        day_start = base_date + timedelta(days=day)
        for origin, dest, duration in ROUTES:
            # 2-3 flights a day per route at different times
            for slot in random.sample(range(6, 22), k=random.choice([2, 3])):
                dep_time = day_start.replace(hour=slot, minute=random.choice([0, 15, 30, 45]))
                arr_time = dep_time + timedelta(minutes=duration)
                flight_id = f"{random.choice(AIRLINES)}{flight_counter}"
                flight_counter += 1
                flights.append({
                    "flight_id": flight_id,
                    "origin": origin,
                    "dest": dest,
                    "dep_time": dep_time,
                    "arr_time": arr_time,
                    "duration_min": duration,
                    "capacity": random.choice([150, 180, 220]),
                })

    return pd.DataFrame(flights)


def find_connecting_flights(schedule: pd.DataFrame, disrupted_flight_id: str,
                             min_connection_min=45, max_connection_min=360) -> pd.DataFrame:
    """
    Given a disrupted flight, finds flights in the schedule that connect
    FROM the disrupted flight's destination within a reasonable connection window.
    These represent passengers' onward journeys that may now be at risk.
    """
    disrupted = schedule[schedule.flight_id == disrupted_flight_id].iloc[0]
    dest = disrupted.dest
    arr = disrupted.arr_time

    candidates = schedule[
        (schedule.origin == dest) &
        (schedule.flight_id != disrupted_flight_id)
    ].copy()

    candidates["connection_min"] = (candidates.dep_time - arr).dt.total_seconds() / 60
    connecting = candidates[
        (candidates.connection_min >= min_connection_min) &
        (candidates.connection_min <= max_connection_min)
    ].sort_values("connection_min")

    return connecting


if __name__ == "__main__":
    G = build_airport_graph()
    print(f"Airport graph: {G.number_of_nodes()} airports, {G.number_of_edges()} routes")

    schedule = generate_flight_schedule(num_days=1)
    print(f"\nGenerated {len(schedule)} flights for the day")
    print(schedule.head())

    # pick a random flight to disrupt
    sample_flight = schedule.iloc[10]
    print(f"\nSimulating disruption on flight {sample_flight.flight_id} "
          f"({sample_flight.origin} -> {sample_flight.dest})")

    connecting = find_connecting_flights(schedule, sample_flight.flight_id)
    print(f"Found {len(connecting)} possible reaccommodation flights onward from {sample_flight.dest}")
    print(connecting[["flight_id", "origin", "dest", "dep_time", "connection_min"]].head())
