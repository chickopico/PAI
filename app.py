# app.py - TRUE ML-POWERED Seattle Airbnb Recommender
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
import folium
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2
import os

# -----------------------------
# 1. Load & Preprocess Data (cached)
# -----------------------------
@st.cache_data
def load_and_prepare_data():
    if not os.path.exists("listings1.csv"):
        st.error("listings1.csv not found! Place it in the same folder as app.py")
        st.stop()
    
    df = pd.read_csv("listings1.csv")
    
    # Basic cleaning
    df['price'] = pd.to_numeric(df['price'].replace(r'[\$,]', '', regex=True), errors='coerce')  # Handle any $ or commas in price
    df = df.dropna(subset=['price', 'latitude', 'longitude'])
    df = df[df['price'] > 20]
    df = df[df['price'].between(30, 1500)]  # Remove $10 scams & $10k mansions
    price_99 = df['price'].quantile(0.99)
    df['price'] = df['price'].clip(upper=price_99)
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df['number_of_reviews_ltm'] = df['number_of_reviews_ltm'].fillna(0)
    df['calculated_host_listings_count'] = df['calculated_host_listings_count'].fillna(1)
    
    # Haversine distance to downtown
    downtown_lat, downtown_lon = 47.6062, -122.3321
    def haversine(lat, lon):
        R = 6371
        dlat = radians(downtown_lat - lat)
        dlon = radians(downtown_lon - lon)
        a = sin(dlat/2)**2 + cos(radians(lat)) * cos(radians(downtown_lat)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    
    df['dist_to_downtown_km'] = df.apply(lambda row: haversine(row['latitude'], row['longitude']), axis=1)
    
    return df

df = load_and_prepare_data()

# -----------------------------
# 2. Train CatBoost ML Model with Evaluation (cached)
# -----------------------------
@st.cache_resource
def train_ml_model(_df):
    features_df = _df.copy()
    
    # Create realistic booking proxy: how often this listing gets booked
    # High reviews in last 12 months + low availability = frequently booked
    features_df['booking_proxy'] = (
        features_df['number_of_reviews_ltm'] /
        (features_df['availability_365'] + 10)  # +10 to avoid division by zero
    )
    features_df['target'] = np.log1p(features_df['booking_proxy'])
    
    # Features for ML (added more for better accuracy: minimum_nights, reviews_per_month)
    X = features_df[[
        'price', 'dist_to_downtown_km', 'number_of_reviews', 'reviews_per_month',
        'calculated_host_listings_count', 'availability_365', 'minimum_nights',
        'room_type', 'neighbourhood_group'
    ]]
    y = features_df['target']
    
    cat_features = ['room_type', 'neighbourhood_group']
    
    # Train-test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = CatBoostRegressor(
        iterations=1000,  # Increased iterations for better learning
        learning_rate=0.03,  # Slightly lower LR for stability
        depth=7,  # Slightly deeper trees
        loss_function='RMSE',
        random_seed=42,
        verbose=0,
        cat_features=cat_features
    )
    
    model.fit(X_train, y_train)
    
    # Predictions and metrics
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    return model, rmse, r2

st.info("Training Machine Learning model (CatBoost)... This takes ~10-20 seconds first time only.")
ml_model, rmse, r2 = train_ml_model(df)

# -----------------------------
# 3. Generate ML + Heuristic Hybrid Score
# -----------------------------
@st.cache_data
def generate_smart_scores(_df, _model):
    data = _df.copy()
    
    # 1. ML Score (learned booking likelihood)
    X_ml = data[[
        'price', 'dist_to_downtown_km', 'number_of_reviews', 'reviews_per_month',
        'calculated_host_listings_count', 'availability_365', 'minimum_nights',
        'room_type', 'neighbourhood_group'
    ]]
    data['ML_Score_Raw'] = _model.predict(X_ml)
    scaler = MinMaxScaler()
    data['ML_Score'] = scaler.fit_transform(data[['ML_Score_Raw']]) * 100
    
    # 2. Heuristic Score (improved: added availability penalty for over-available listings)
    data['price_score'] = 100 - scaler.fit_transform(data[['price']]) * 100
    data['review_score'] = scaler.fit_transform(data[['number_of_reviews']]) * 100
    data['location_score'] = 100 - scaler.fit_transform(data[['dist_to_downtown_km']]) * 80
    data['location_score'] = data['location_score'].clip(20, 100)
    data['availability_penalty'] = scaler.fit_transform(data[['availability_365']]) * -20  # Penalize high availability
    
    room_bonus = {'Entire home/apt': 20, 'Private room': 10, 'Hotel room': 15, 'Shared room': 0}
    data['room_bonus'] = data['room_type'].map(room_bonus).fillna(5)
    
    data['Heuristic_Score'] = (
        data['price_score'] * 0.30 +
        data['review_score'] * 0.25 +
        data['location_score'] * 0.20 +
        data['room_bonus'] * 0.15 +
        data['availability_penalty'] * 0.10
    )
    
    # 3. FINAL HYBRID SCORE (Adjusted weights: more emphasis on ML since it's data-driven)
    data['Smart_Score'] = 0.7 * data['ML_Score'] + 0.3 * data['Heuristic_Score']
    
    return data

df = generate_smart_scores(df, ml_model)

# -----------------------------
# 4. Streamlit App UI (Improved: Added filters, better display, model metrics)
# -----------------------------
st.title("Seattle Airbnb – ML-Powered Best Value Finder")
st.success(f"Machine Learning Model Trained! CatBoost R² Score: {r2:.2f} | RMSE: {rmse:.2f} (Higher R² means better predictive accuracy)")

col1, col2, col3 = st.columns(3)
with col1:
    budget = st.slider("Max Budget ($/night)", 50, 800, 250)
with col2:
    min_reviews = st.slider("Min Reviews", 0, 1000, 50)
with col3:
    max_dist = st.slider("Max Distance to Downtown (km)", 1, 20, 5)

neighbourhoods = ['All'] + sorted(df['neighbourhood'].unique().tolist())
selected_neigh = st.multiselect("Neighborhood", neighbourhoods, default=['All'])

room_types = ['All'] + sorted(df['room_type'].unique().tolist())
selected_room = st.selectbox("Room Type", room_types, index=0)

top_n = st.slider("Show Top", 5, 50, 10)  # Increased max for more options

# Apply filters (Added distance filter)
filtered = df.copy()
if 'All' not in selected_neigh:
    filtered = filtered[filtered['neighbourhood'].isin(selected_neigh)]
if selected_room != 'All':
    filtered = filtered[filtered['room_type'] == selected_room]
filtered = filtered[
    (filtered['price'] <= budget) &
    (filtered['number_of_reviews'] >= min_reviews) &
    (filtered['dist_to_downtown_km'] <= max_dist)
]

results = filtered.sort_values('Smart_Score', ascending=False).head(top_n)

st.subheader(f"Top {len(results)} Best Places Right Now (ML + Expert Score)")
display = results[['name', 'neighbourhood', 'room_type', 'price',
                   'number_of_reviews', 'Smart_Score', 'dist_to_downtown_km']].round(2)
display['Smart_Score'] = display['Smart_Score'].astype(float).round(1)
st.dataframe(display, use_container_width=True, hide_index=True)

# Map (Improved: Added clustering for better visualization if many points)
if not results.empty:
    st.subheader("Map of Top Recommendations")
    from folium.plugins import MarkerCluster
    m = folium.Map(location=[47.6062, -122.3321], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(m)
    for _, r in results.iterrows():
        folium.Marker(
            location=[r.latitude, r.longitude],
            popup=f"<b>{r['name'][:60]}</b><br>Price: ${r.price}<br>Score: {r.Smart_Score:.1f}<br>Reviews: {int(r.number_of_reviews)}",
            icon=folium.Icon(color='red' if r.Smart_Score > 80 else 'orange' if r.Smart_Score > 70 else 'blue')
        ).add_to(marker_cluster)
    st_folium(m, width=700, height=500)


st.caption("ML Model: CatBoost (Trained with 80/20 split) | Hybrid Score = 70% ML + 30% Expert Rules | Nov 2025 Data")
