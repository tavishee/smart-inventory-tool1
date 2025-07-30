# Smart Inventory Optimization Tool – v4 with Demand Map

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# Page Setup
st.set_page_config(page_title="Smart Inventory Optimization Tool", layout="wide")
st.title("🚗 Smart Inventory Optimization Tool – v4")

# Sidebar - File Upload
st.sidebar.header("📁 Upload Inventory Data File")
uploaded_file = st.sidebar.file_uploader("Upload an Excel or CSV file", type=["xlsx", "csv"])

# Main App
if uploaded_file:
    # Read File
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📊 Uploaded Inventory Data")
    st.dataframe(df)

    # ---------- Feature 1: Automated Relocation Engine ----------
    st.markdown("### 🚚 Relocation Profit Analysis")
    cost_per_km = st.sidebar.number_input("Transport Cost per km (INR)", 10, 100, 25)
    other_costs = st.sidebar.number_input("Other Logistics Cost per Car (INR)", 1000, 10000, 3000)

    if {'source_city', 'dest_city', 'distance_km', 'expected_profit'}.issubset(df.columns):
        df['relocation_cost'] = df['distance_km'] * cost_per_km + other_costs
        df['net_gain'] = df['expected_profit'] - df['relocation_cost']
        st.write("Suggested Relocations (Only Profitable Ones):")
        st.dataframe(df[df['net_gain'] > 0][['car_id', 'source_city', 'dest_city', 'net_gain']])
    else:
        st.warning("Columns required: source_city, dest_city, distance_km, expected_profit")

    # ---------- Feature 2: Market Gap Purchase Suggestions ----------
    st.markdown("### 🛒 Purchase Suggestions Based on Market Gaps")
    if {'city', 'car_type', 'demand', 'supply'}.issubset(df.columns):
        market_gap = df.groupby(['city', 'car_type']).agg({'demand':'sum', 'supply':'sum'}).reset_index()
        market_gap['gap'] = market_gap['demand'] - market_gap['supply']
        st.write("High-Gap Segments:")
        st.dataframe(market_gap[market_gap['gap'] > 0].sort_values('gap', ascending=False))
    else:
        st.warning("Columns required: city, car_type, demand, supply")

    # ---------- Feature 3: Inventory Risk Dashboard ----------
    st.markdown("### ⚠️ Inventory Risk Dashboard")
    if 'days_in_inventory' in df.columns:
        st.write("Cars older than 45 days in inventory:")
        st.dataframe(df[df['days_in_inventory'] > 45][['car_id', 'city', 'days_in_inventory']].sort_values('days_in_inventory', ascending=False))
    else:
        st.warning("Column required: days_in_inventory")

    # ---------- Feature 4: Price Optimization ----------
    st.markdown("### 📈 Demand Forecasting & Price Optimization")
    if {'past_demand', 'days_on_platform'}.issubset(df.columns):
        X = df[['past_demand', 'days_on_platform']]
        if 'expected_price' in df.columns:
            y = df['expected_price']
        elif 'price' in df.columns:
            y = df['price'] * 1.05
        else:
            st.warning("Need either 'expected_price' or 'price' column")
            y = None

        if y is not None:
            model = GradientBoostingRegressor()
            model.fit(X, y)
            df['predicted_price'] = model.predict(X)
            st.write("Predicted Optimal Prices:")
            st.dataframe(df[['car_id', 'city', 'predicted_price']])
    else:
        st.warning("Columns required: past_demand, days_on_platform")

    # ---------- Feature 5: 🌍 City-wise Demand Map (Interactive) ----------
    st.markdown("### 🌍 City-wise Demand Map (Interactive)")

    if {'city', 'demand'}.issubset(df.columns):
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            city_coords = {
                "Delhi": (28.6139, 77.2090),
                "Mumbai": (19.0760, 72.8777),
                "Bangalore": (12.9716, 77.5946),
                "Chennai": (13.0827, 80.2707),
                "Hyderabad": (17.3850, 78.4867),
                "Pune": (18.5204, 73.8567),
                "Kolkata": (22.5726, 88.3639),
                "Jaipur": (26.9124, 75.7873),
                "Ahmedabad": (23.0225, 72.5714),
                "Lucknow": (26.8467, 80.9462)
            }
            df['latitude'] = df['city'].map(lambda x: city_coords.get(x, (None, None))[0])
            df['longitude'] = df['city'].map(lambda x: city_coords.get(x, (None, None))[1])

        map_df = df[['city', 'demand', 'latitude', 'longitude']].dropna()
        map_df = map_df.groupby(['city', 'latitude', 'longitude'])['demand'].sum().reset_index()

        import plotly.express as px
        fig = px.scatter_geo(
            map_df,
            lat='latitude',
            lon='longitude',
            text='city',
            size='demand',
            color='demand',
            projection='natural earth',
            title='Demand by City',
            color_continuous_scale='YlOrRd'
        )
        fig.update_layout(geo_scope='asia')
        st.plotly_chart(fig)
    else:
        st.warning("Columns required: city, demand (plus optional: latitude, longitude)")

else:
    st.info("Upload a valid inventory file to get started.")
