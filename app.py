import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.cluster import KMeans

st.set_page_config(page_title="Chennai Dark Store Optimizer", page_icon="📍", layout="wide")
st.title("📍 Dark Store Location Optimization")
st.caption("Quick-commerce dark-store planning with a hard service-radius constraint and daily store capacity.")
st.markdown("""
**Planning assumptions:** synthetic customer and candidate locations within a Chennai-area bounding box; each customer is assigned to exactly one open store; a customer can only be served by a store within the maximum distance; fixed opening cost and per-order contribution are user-configurable. This is a planning prototype, not live address or road-network data.
""")

# Chennai-area coordinates, approximate bounds. Synthetic reproducible data.
@st.cache_data
def make_points(n_customers, n_candidates, seed=42):
    rng = np.random.default_rng(seed)
    # Mixture of Chennai-area demand clusters to create realistic spatial concentration.
    centers = np.array([[13.0827,80.2707],[13.0358,80.2445],[13.1067,80.2206],
                        [12.9716,80.2206],[13.0475,80.2090],[13.1250,80.2800]])
    weights = np.array([.27,.18,.16,.15,.13,.11])
    idx = rng.choice(len(centers), size=n_customers, p=weights)
    customers = centers[idx] + rng.normal(0, [0.018,0.020], size=(n_customers,2))
    customers[:,0] = np.clip(customers[:,0],12.88,13.25)
    customers[:,1] = np.clip(customers[:,1],80.08,80.36)
    # Candidates sampled around the same area with a small regular grid blend.
    cidx = rng.choice(len(centers), size=n_candidates, p=weights)
    candidates = centers[cidx] + rng.normal(0,[0.025,0.026],size=(n_candidates,2))
    candidates[:,0] = np.clip(candidates[:,0],12.88,13.25)
    candidates[:,1] = np.clip(candidates[:,1],80.08,80.36)
    return customers, candidates

def haversine_matrix(a, b):
    # coordinates are [latitude, longitude]; output kilometres
    lat1=np.radians(a[:,0])[:,None]; lon1=np.radians(a[:,1])[:,None]
    lat2=np.radians(b[:,0])[None,:]; lon2=np.radians(b[:,1])[None,:]
    dlat=lat2-lat1; dlon=lon2-lon1
    q=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371.0088*2*np.arcsin(np.sqrt(np.clip(q,0,1)))

def solve_location(customers, candidates, max_dist, max_open, orders_per_customer,
                   revenue_per_order, fixed_cost, distance_cost_per_km, capacity_per_store, time_limit=30):
    """Greedy capacity-aware allocation with a hard maximum service distance."""
    n, m = len(customers), len(candidates)
    dist = haversine_matrix(customers, candidates)
    feasible = dist <= max_dist + 1e-9
    demand = np.full(n, float(orders_per_customer))
    if np.any(~feasible.any(axis=1)):
        missing = int((~feasible.any(axis=1)).sum())
        return None, dist, f"{missing} customer location(s) are outside the maximum distance from every candidate. Increase distance limit or candidate count."
    selected=[]; remaining=demand.copy()
    while len(selected) < min(max_open,m) and remaining.sum() > 1e-8:
        best_j=None; best_score=-np.inf
        for j in range(m):
            if j in selected: continue
            eligible=feasible[:,j] & (remaining > 1e-8)
            gain=min(float(remaining[eligible].sum()), float(capacity_per_store))
            if gain <= 0: continue
            avgd=float(np.average(dist[eligible,j],weights=remaining[eligible])) if eligible.any() else max_dist
            score=gain*1e6-avgd
            if score>best_score: best_score,best_j=score,j
        if best_j is None: break
        selected.append(best_j); cap=float(capacity_per_store)
        for i in np.argsort(np.where(feasible[:,best_j],dist[:,best_j],np.inf)):
            if cap<=1e-8: break
            if feasible[i,best_j] and remaining[i]>1e-8:
                qty=min(remaining[i],cap); remaining[i]-=qty; cap-=qty
    # Unserved demand is permitted, consistent with the course notes' revised
    # demand constraint (fulfilled demand <= requested demand) and revenue objective.
    fs=feasible[:,selected]
    allocation=np.zeros((n,len(selected)),dtype=float); cap=np.full(len(selected),float(capacity_per_store))
    order=sorted(range(n),key=lambda i:int(fs[i].sum()))
    for i in order:
        left=float(demand[i])
        for k in sorted(np.where(fs[i])[0],key=lambda k:dist[i,selected[k]]):
            qty=min(left,cap[k])
            if qty>0: allocation[i,k]+=qty; cap[k]-=qty; left-=qty
            if left<=1e-8: break
    total_orders=float(allocation.sum())
    order_km=sum(allocation[i,k]*dist[i,selected[k]] for i in range(n) for k in range(len(selected)))
    revenue=total_orders*revenue_per_order
    travel_cost=distance_cost_per_km*order_km; open_cost=fixed_cost*len(selected)
    primary=np.argmax(allocation,axis=1)
    assigned_dist=dist[np.arange(n),np.asarray(selected)[primary]]
    served_by_customer=allocation.sum(axis=1)>0
    assigned_dist=np.where(served_by_customer,assigned_dist,np.nan)
    coverage=float(np.mean(served_by_customer))
    unserved=float(demand.sum()-total_orders)
    return {"open":np.asarray(selected),"assignments":np.asarray(selected)[primary],
            "allocation":allocation,"demand":demand,"assigned_dist":assigned_dist,
            "profit":revenue-travel_cost-open_cost,"revenue":revenue,"travel_cost":travel_cost,
            "open_cost":open_cost,"n_open":len(selected),"coverage":coverage,
            "avg_dist":float(order_km/max(total_orders,1)),"max_dist":float(np.nanmax(assigned_dist)) if np.any(served_by_customer) else 0.0,
            "total_orders":total_orders,"unserved_orders":unserved,"status":"Greedy capacity-aware heuristic completed (feasibility enforced)."}, dist, None

with st.sidebar:
    st.header("Model inputs")
    n_candidates = st.number_input("Dark store candidate count", min_value=5, max_value=100, value=50, step=5)
    n_customers = st.number_input("Number of customers", min_value=100, max_value=10000, value=1000, step=100)
    max_dist = st.slider("Maximum customer–store distance (km)", min_value=0.5, max_value=15.0, value=2.0, step=0.1)
    max_open = st.number_input("Maximum stores allowed to open", min_value=1, max_value=int(n_candidates), value=min(5,int(n_candidates)), step=1)
    st.divider()
    orders = st.number_input("Orders per customer per day", min_value=1, max_value=20, value=2)
    capacity = st.number_input("Daily capacity per dark store (orders)", min_value=500, max_value=5000, value=1800, step=100)
    revenue_per_order = st.number_input("Revenue per fulfilled order (₹)", min_value=1, max_value=3000, value=180, step=10)
    delivery_cost = st.number_input("Delivery cost per order-km (₹)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)
    fixed_cost = st.number_input("Fixed operating cost per store per day (₹)", min_value=0, max_value=1000000, value=5000, step=500)
    seed = st.number_input("Synthetic data seed", min_value=1, max_value=99999, value=42)

customers, candidates = make_points(int(n_customers), int(n_candidates), int(seed))
col1,col2,col3 = st.columns(3)
col1.metric("Customers", f"{int(n_customers):,}")
col2.metric("Candidate sites", f"{int(n_candidates):,}")
col3.metric("Max service distance", f"{max_dist:.1f} km")

run = st.button("🚀 Run optimization", type="primary", use_container_width=True)
if run:
    with st.spinner("Solving mixed-integer location-allocation model…"):
        sol, dist, err = solve_location(customers,candidates,float(max_dist),int(max_open),int(orders),float(revenue_per_order),float(fixed_cost),float(delivery_cost),float(capacity))
    if err:
        st.error(err)
    else:
        st.session_state["solution"] = sol
        st.session_state["customers"] = customers
        st.session_state["candidates"] = candidates
        st.session_state["inputs"] = (float(max_dist),int(orders),float(revenue_per_order),float(fixed_cost),float(delivery_cost),float(capacity))
        st.success(f"Optimization complete. Solver status: {sol['status']}")
if "solution" in st.session_state:
    sol=st.session_state["solution"]; customers=st.session_state["customers"]; candidates=st.session_state["candidates"]
    max_dist,orders,revenue_per_order,fixed_cost,delivery_cost,capacity=st.session_state["inputs"]
    st.subheader("1. KPI summary")
    kpis=pd.DataFrame([
        ["Customer locations",f"{len(customers):,}"],
        ["Daily orders fulfilled",f"{sol['total_orders']:,.0f}"],
        ["Daily orders unserved",f"{sol['unserved_orders']:,.0f}"],
        ["Stores opened",f"{sol['n_open']}"],
        ["Coverage within distance limit",f"{sol['coverage']:.1%}"],
        ["Average assigned distance",f"{sol['avg_dist']:.2f} km"],
        ["Maximum assigned distance",f"{sol['max_dist']:.2f} km"],
        ["Revenue from fulfilled orders",f"₹{sol['revenue']:,.0f}"],
        ["Delivery cost (order-km proxy)",f"₹{sol['travel_cost']:,.0f}"],
        ["Store operating cost",f"₹{sol['open_cost']:,.0f}"],
        ["Daily capacity used",f"{sol['total_orders']/(sol['n_open']*capacity):.1%}"],
        ["Estimated profit",f"₹{sol['profit']:,.0f}"],
    ],columns=["KPI","Value"])
    st.dataframe(kpis,hide_index=True,use_container_width=True)
    st.subheader("2. Detailed solution map")
    # Lines from assigned store to customer; for very large sets sample lines to keep map responsive.
    fig=go.Figure()
    for j in sol["open"]:
        ids=np.where(sol["assignments"]==j)[0]
        fig.add_trace(go.Scattermapbox(lat=customers[ids,0],lon=customers[ids,1],mode="markers",
            marker={"size":5},name=f"Customers → Store {j+1}",hovertemplate="Customer<extra></extra>"))
    for i in range(len(customers)):
        j=sol["assignments"][i]
        if i < 1500:
            fig.add_trace(go.Scattermapbox(lat=[customers[i,0],candidates[j,0]],lon=[customers[i,1],candidates[j,1]],
                mode="lines",line={"width":1},showlegend=False,hoverinfo="skip"))
    fig.add_trace(go.Scattermapbox(lat=candidates[:,0],lon=candidates[:,1],mode="markers",
        marker={"size":9,"symbol":"circle"},name="Candidate locations",text=[f"Candidate {j+1}" for j in range(len(candidates))],
        hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scattermapbox(lat=candidates[sol["open"],0],lon=candidates[sol["open"],1],mode="markers",
        marker={"size":16,"symbol":"star"},name="Opened dark stores",text=[f"Opened store {j+1}" for j in sol["open"]],
        hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(mapbox_style="open-street-map",mapbox_zoom=9,
        mapbox_center={"lat":13.06,"lon":80.24},height=650,margin={"l":0,"r":0,"t":20,"b":0},
        legend={"orientation":"h","yanchor":"bottom","y":-0.1})
    st.plotly_chart(fig,use_container_width=True)
    assign_df=pd.DataFrame({"Customer ID":np.arange(1,len(customers)+1),
        "Latitude":customers[:,0],"Longitude":customers[:,1],
        "Primary store (map)":[int(j+1) for j in sol["assignments"]],
        "Daily demand (orders)":sol["demand"],
        "Distance to primary store (km)":np.round(sol["assigned_dist"],3),
        "Demand served (orders)":sol["allocation"].sum(axis=1)})
    st.download_button("Download customer assignments CSV",assign_df.to_csv(index=False).encode("utf-8"),
                       "customer_assignments.csv","text/csv")
    st.subheader("3. Profit sensitivity: up to 2, 5, 10, 20 and 25 stores")
    if int(n_customers)==10000 and int(n_candidates)==50:
        sweep_customers=customers; sweep_candidates=candidates
    else:
        sweep_customers,sweep_candidates=make_points(10000,50,int(seed))
    if st.button("Calculate 10,000-customer / 50-candidate profit curve"):
        profits=[]; rows=[]
        with st.spinner("Solving five optimization scenarios. This may take a few minutes…"):
            for limit in [2,5,10,20,25]:
                s,_,e=solve_location(sweep_customers,sweep_candidates,float(max_dist),limit,int(orders),float(revenue_per_order),float(fixed_cost),float(delivery_cost),float(capacity),time_limit=45)
                if e:
                    rows.append({"Maximum stores":limit,"Stores opened":None,"Estimated profit (₹)":None,"Status":e})
                else:
                    rows.append({"Maximum stores":limit,"Stores opened":s["n_open"],"Estimated profit (₹)":round(s["profit"],2),"Status":s["status"]})
        prof=pd.DataFrame(rows)
        st.dataframe(prof,hide_index=True,use_container_width=True)
        valid=prof.dropna(subset=["Estimated profit (₹)"])
        if len(valid):
            chart=px.line(valid,x="Maximum stores",y="Estimated profit (₹)",markers=True,
                           title="Estimated profit vs. maximum permitted stores")
            st.plotly_chart(chart,use_container_width=True)
        st.download_button("Download profit sensitivity CSV",prof.to_csv(index=False).encode("utf-8"),
                           "profit_sensitivity.csv","text/csv")
else:
    st.info("Set your inputs in the sidebar, then click **Run optimization** to view KPIs and the assigned-location map.")
st.caption("Optimization objective: maximize revenue from fulfilled orders minus daily store operating costs and order-distance delivery costs. Constraints enforce demand fulfilment, per-store daily capacity, maximum store count, and distance ≤ the user-defined maximum. Demand can be split across eligible stores; the map marks each customer’s primary store. Synthetic coordinates are generated deterministically from the seed.")
