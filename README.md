# ✈️ Flight Disruption Rebooking Optimizer

**Live demo:** https://flightdisruptionoptimizer-npzbdp3jkbnrj6dwbtnxfz.streamlit.app/

An end-to-end system that automatically reaccommodates passengers after a
flight disruption (delay/cancellation) — combining **machine learning**
(predicting which passengers should be prioritized) with **operations
research** (optimally assigning limited seats on rebooking flights).

This mirrors a real airline revenue-management / ops problem: when a flight
is disrupted, an airline has to quickly decide who gets rebooked on which
onward flight, under hard seat constraints, while balancing fairness,
contractual obligations (fare class), and loyalty commitments.

## Why this project

Most student ML projects stop at "train a model, report accuracy." This one
goes a step further: the model's output isn't the end product — it's an
**input to a downstream decision system** (a constrained optimizer), which
is much closer to how ML is actually used in production at companies like
airlines and travel-tech platforms.

## How it works

```
Flight network (networkx)
        │
        ▼
Simulate a disruption on a chosen flight
        │
        ▼
Find candidate onward ("rebooking") flights
        │
        ▼
Simulate the disrupted flight's passenger manifest
  (fare class, loyalty tier, connection tightness, group size, etc.)
        │
        ▼
XGBoost predicts each passenger's rebooking PRIORITY score
        │
        ▼
OR-Tools (CP-SAT) solves the optimal seat assignment,
maximizing total priority served under seat-capacity constraints
        │
        ▼
Streamlit UI shows the disruption, priority scores, and final rebooking plan
```

### 1. Flight network — `src/network.py`
Builds a small airport/route graph (`networkx`) and generates a realistic
daily flight schedule. Given a disrupted flight, finds valid onward
connections within a realistic connection-time window.

### 2. Passenger simulation — `src/passengers.py`
Generates a synthetic passenger manifest per flight with airline-realistic
features (fare class, loyalty tier, booking lead time, group size,
connection buffer, historical no-show rate). Since proprietary airline
booking data isn't publicly available, priority labels are generated via a
rule-based simulation with noise — capturing realistic patterns (higher fare
class / loyalty / tighter connections / larger groups → higher priority)
without being a trivial lookup table.

### 3. ML model — `src/model.py`
Trains an **XGBoost classifier** on ~23,000 simulated passengers across 150
disrupted-flight scenarios to predict rebooking priority. Achieves **ROC-AUC
≈ 0.92** on held-out data. Feature importances confirm the model learns
sensible patterns (fare class and loyalty tier dominate, followed by
connection risk).

### 4. Optimizer — `src/optimizer.py`
Uses **Google OR-Tools' CP-SAT solver** to solve a constrained assignment
problem: each passenger can be assigned to at most one flight, each flight
has limited available seats, and the objective maximizes total priority
served (weighted toward earlier/better connecting flights). This is a real
combinatorial optimization problem, not a greedy heuristic.

### 5. UI — `app/streamlit_app.py`
Lets you pick any flight in the simulated schedule to "disrupt," runs the
full pipeline live, and visualizes the rebooking outcome by fare class,
priority distribution, and a full passenger-level rebooking table.

## Running locally

```bash
git clone <your-repo-url>
cd disruption_optimizer
pip install -r requirements.txt
cd src && python model.py        # trains and saves the model (~1-2 min)
cd ../app && streamlit run streamlit_app.py
```

## Tech stack

Python · networkx · pandas/numpy · XGBoost · scikit-learn · OR-Tools (CP-SAT) · Streamlit · Plotly

## What I'd improve with more time

- Replace the rule-based priority simulation with real historical ops data if available
- Add cascading disruption effects (one delay triggering downstream delays)
- Model crew scheduling constraints alongside passenger rebooking
- A/B test the optimizer's objective function against a simple greedy baseline to quantify the improvement

---
*All data in this project is synthetically generated for demonstration purposes. No real airline, passenger, or proprietary data is used.*
