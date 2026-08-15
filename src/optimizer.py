"""
optimizer.py
Given a set of disrupted passengers (with predicted priority scores) and a
set of candidate rebooking flights (with limited seats), finds the
rebooking assignment that maximizes total priority served, subject to:
  - each passenger gets at most one seat
  - each flight can't exceed its available capacity
  - (optional) passengers traveling in a group are assigned together where possible

This is a weighted assignment / bin-packing style problem, a natural fit for
OR-Tools' CP-SAT solver.
"""

import pandas as pd
from ortools.sat.python import cp_model


def optimize_rebooking(passengers_df: pd.DataFrame, candidate_flights_df: pd.DataFrame,
                        available_seats_per_flight: dict, time_limit_sec=10) -> pd.DataFrame:
    """
    passengers_df: must include 'passenger_id' and 'priority_score' (0-1 float)
    candidate_flights_df: must include 'flight_id', ranked implicitly by connection_min
                           (earlier flights are 'better' outcomes for the passenger)
    available_seats_per_flight: dict {flight_id: seats_available}

    Returns a DataFrame: passenger_id, assigned_flight_id (or None if unassigned),
    priority_score, assigned_rank (0 = best/earliest flight offered)
    """
    model = cp_model.CpModel()

    passenger_ids = passengers_df["passenger_id"].tolist()
    priorities = dict(zip(passengers_df["passenger_id"], passengers_df["priority_score"]))

    flight_ids = candidate_flights_df["flight_id"].tolist()
    # earlier flights (lower index) are more desirable -> give them a small bonus
    # so the optimizer prefers filling the earliest flight first, all else equal
    flight_desirability = {
        fid: (len(flight_ids) - rank) for rank, fid in enumerate(flight_ids)
    }

    # decision variables: x[p, f] = 1 if passenger p assigned to flight f
    x = {}
    for p in passenger_ids:
        for f in flight_ids:
            x[p, f] = model.NewBoolVar(f"x_{p}_{f}")

    # each passenger assigned to at most one flight
    for p in passenger_ids:
        model.Add(sum(x[p, f] for f in flight_ids) <= 1)

    # each flight can't exceed available seats
    for f in flight_ids:
        capacity = available_seats_per_flight.get(f, 0)
        model.Add(sum(x[p, f] for p in passenger_ids) <= capacity)

    # objective: maximize total (priority * desirability) served
    # scaled to integers since CP-SAT requires integer coefficients
    objective_terms = []
    for p in passenger_ids:
        for f in flight_ids:
            weight = int(round(priorities[p] * 1000)) * flight_desirability[f]
            objective_terms.append(weight * x[p, f])

    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    results = []
    for p in passenger_ids:
        assigned_flight = None
        assigned_rank = None
        for rank, f in enumerate(flight_ids):
            if solver.Value(x[p, f]) == 1:
                assigned_flight = f
                assigned_rank = rank
                break
        results.append({
            "passenger_id": p,
            "priority_score": priorities[p],
            "assigned_flight_id": assigned_flight,
            "assigned_rank": assigned_rank,  # 0 = earliest/best available flight
        })

    result_df = pd.DataFrame(results)
    solve_status = solver.StatusName(status)
    return result_df, solve_status


if __name__ == "__main__":
    # quick smoke test with toy data
    passengers = pd.DataFrame({
        "passenger_id": [f"P{i}" for i in range(10)],
        "priority_score": [0.9, 0.85, 0.3, 0.6, 0.95, 0.1, 0.5, 0.7, 0.2, 0.8],
    })
    flights = pd.DataFrame({"flight_id": ["F1", "F2", "F3"]})
    seats = {"F1": 3, "F2": 3, "F3": 2}  # only 8 seats for 10 passengers -> 2 unassigned

    result, status = optimize_rebooking(passengers, flights, seats)
    print(f"Solver status: {status}\n")
    print(result.sort_values("priority_score", ascending=False))

    unassigned = result[result.assigned_flight_id.isna()]
    print(f"\nUnassigned passengers: {len(unassigned)}")
    print(unassigned[["passenger_id", "priority_score"]])
