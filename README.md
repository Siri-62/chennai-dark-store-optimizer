# Chennai Dark Store Location Optimizer

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Upload `app.py` and `requirements.txt` to the deployment service/repository supported by your Streamline workspace. A live deployment requires access to your Streamline account; this package does not itself create a hosted URL.

## Model
The application uses a scalable greedy capacity-aware location-allocation heuristic. Each customer location has daily demand, which may be split among eligible stores. Only stores within the maximum straight-line distance are eligible. Each store has a daily order capacity, and the number of open stores is bounded by the selected limit. Unserved demand is permitted and reported, consistent with the course notes' revised demand-satisfaction formulation. The objective maximizes revenue from fulfilled orders minus delivery cost (order-kilometre proxy) and daily store operating costs.

## Assumptions and limitations
- Chennai-area customer and candidate locations are synthetic and reproducible using the seed; they are not real addresses or customer data.
- Distance is straight-line haversine distance, not road distance or actual delivery time.
- Daily customer demand is uniform and user-configurable; the model does not include differing demand profiles or store-specific costs.
- Default service radius is 2 km and default daily store capacity is 1,800 orders, based on the ranges in the supplied course notes.
- The profit sensitivity analysis uses 10,000 customer locations and 50 candidate sites for limits of 2, 5, 10, 20, and 25 stores.
- This is a heuristic, not a proof of global optimality.
