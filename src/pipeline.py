"""
pipeline.py
End-to-end orchestration:
  1. Build flight network + today's schedule
  2. Pick/receive a disrupted flight
  3. Find candidate onward flights for reaccommodation
  4. Simulate the disrupted flight's passenger manifest
  5. Predict rebooking priority for each passenger (XGBoost)
  6. Solve the optimal rebooking assignment (OR-Tools)
  7. Return a clean summary for display in the UI
"""

import pandas as pd
from network import build_airport_graph, generate_flight_schedule, find_connecting_flights
from passengers import generate_passengers
from model import load_model, predict_priority
from optimizer import optimize_rebooking


def run_disruption_scenario(schedule: pd.DataFrame, disrupted_flight_id: str,
                             num_passengers: int, model, encoders,
                             seed=None):
    disrupted = schedule[schedule.flight_id == disrupted_flight_id].iloc[0]

    connecting = find_connecting_flights(schedule, disrupted_flight_id)
    conn_available = len(connecting) > 0

    passengers = generate_passengers(
        flight_id=disrupted_flight_id,
        num_passengers=num_passengers,
        connecting_flight_available=conn_available,
        seed=seed,
    )

    priority_scores = predict_priority(passengers, model, encoders)
    passengers["priority_score"] = priority_scores

    if len(connecting) == 0:
        return {
            "disrupted_flight": disrupted.to_dict(),
            "passengers": passengers,
            "connecting_flights": connecting,
            "assignment": None,
            "solve_status": "NO_CONNECTING_FLIGHTS",
        }

    # naive seat availability assumption for the simulation: each candidate
    # flight has some free seats independent of the disrupted flight's own pax
    seats_available = {
        row.flight_id: int(row.capacity * 0.25)  # assume ~25% of seats free, a realistic load factor buffer
        for row in connecting.itertuples()
    }

    assignment, status = optimize_rebooking(
        passengers[["passenger_id", "priority_score"]],
        connecting[["flight_id"]],
        seats_available,
    )

    return {
        "disrupted_flight": disrupted.to_dict(),
        "passengers": passengers,
        "connecting_flights": connecting,
        "seats_available": seats_available,
        "assignment": assignment,
        "solve_status": status,
    }


if __name__ == "__main__":
    print("Loading trained model...")
    model, encoders = load_model()

    print("Building schedule...")
    schedule = generate_flight_schedule(num_days=1)

    # pick a flight that actually has onward connections for a good demo
    best_flight = None
    for fid in schedule.flight_id:
        conn = find_connecting_flights(schedule, fid)
        if len(conn) >= 3:
            best_flight = fid
            break

    print(f"\nRunning scenario for disrupted flight: {best_flight}")
    result = run_disruption_scenario(schedule, best_flight, num_passengers=180, model=model, encoders=encoders, seed=1)

    print(f"\nSolve status: {result['solve_status']}")
    print(f"Total passengers: {len(result['passengers'])}")
    print(f"Candidate onward flights: {len(result['connecting_flights'])}")

    assignment = result["assignment"]
    assigned = assignment[assignment.assigned_flight_id.notna()]
    unassigned = assignment[assignment.assigned_flight_id.isna()]
    print(f"Successfully rebooked: {len(assigned)} / {len(assignment)}")
    print(f"Unassigned (no seats): {len(unassigned)}")

    print("\nTop 10 highest priority passengers and their outcome:")
    top = assignment.sort_values("priority_score", ascending=False).head(10)
    print(top)
