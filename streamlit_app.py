import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("random_forest_tourist_model.pkl")

st.set_page_config(page_title="Tourist Prediction Dashboard", layout="centered")

# Create tabs
tab1, tab2 = st.tabs(["🔮 Predict", "📈 Trends"])

# ---------------------------- TAB 1: PREDICTION ----------------------------
with tab1:
    st.title("🌍 Tourist Prediction Dashboard")
    st.markdown("🧠 **Note:** Model was trained on data up to 2025. Predictions beyond that are based on trend assumptions.")
    st.markdown("---")

    st.subheader("📍 Basic Information")
    country = st.selectbox("Country", ['Bhutan', 'Nepal', 'Thailand', 'India', 'Sri Lanka'])
    region = st.selectbox("Region", ['East', 'West', 'North', 'South', 'Central'])
    month = st.selectbox("Month", [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ])
    year = st.slider("Year", min_value=2015, max_value=2025, value=2023)
    month_num = list(range(1, 13))[["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(month)]

    # Expanders
    with st.expander("🌤️ Weather & Festival Factors"):
        avg_temperature = st.slider("Average Temperature (°C)", 5.0, 45.0, 25.0)
        rainfall_mm = st.slider("Rainfall (mm)", 0.0, 400.0, 100.0)
        festival = st.selectbox("Festival Season?", [True, False])

    with st.expander("🍃 Travel Restrictions & Access"):
        covid_restriction = st.selectbox("Covid Restrictions?", [True, False])
        visa_required = st.selectbox("Visa Required?", [True, False])

    with st.expander("💵 Economic Indicators"):
        gdp_growth_rate = st.slider("GDP Growth Rate (%)", -5.0, 10.0, 3.0)
        foreign_exchange_rate = st.slider("Foreign Exchange Rate", 30.0, 100.0, 70.0)
        internet_mentions = st.slider("Internet Mentions", 0, 10000, 1000)
        global_travel_index = st.slider("Global Travel Index", 0.0, 100.0, 50.0)

    with st.expander("✈️ Travel Cost & Hotel Info"):
        airline_ticket_price = st.slider("Airline Ticket Price ($)", 50.0, 1000.0, 300.0)
        hotel_occupancy_rate = st.slider("Hotel Occupancy Rate (%)", 0.0, 100.0, 60.0)

    # Predict
    if st.button("🎯 Predict Number of Tourists"):
        input_df = pd.DataFrame({
            'country': [country],
            'region': [region],
            'year': [year],
            'month': [month_num],
            'avg_temperature': [avg_temperature],
            'rainfall_mm': [rainfall_mm],
            'festival': [festival],
            'covid_restriction': [covid_restriction],
            'gdp_growth_rate': [gdp_growth_rate],
            'foreign_exchange_rate': [foreign_exchange_rate],
            'internet_mentions': [internet_mentions],
            'global_travel_index': [global_travel_index],
            'airline_ticket_price': [airline_ticket_price],
            'hotel_occupancy_rate': [hotel_occupancy_rate],
            'visa_required': [visa_required],
        })

        predicted_tourists = model.predict(input_df)[0]
        estimated_revenue = predicted_tourists * np.random.uniform(100, 300)

        st.markdown("---")
        st.subheader("📊 Prediction Summary")

        # Big number display
        st.markdown(f"""
        <div style='text-align: center; padding: 10px 0;'>
            <h1 style='font-size: 64px; margin-bottom: 0;'>🧍 {int(predicted_tourists):,}</h1>
            <p style='font-size: 20px;'>Estimated Tourists in <strong>{country}</strong> during <strong>{month} {year}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='text-align: center; padding: 10px 0;'>
            <h2 style='font-size: 48px; margin-bottom: 0;'>💵 ${int(estimated_revenue):,}</h2>
            <p style='font-size: 18px;'>Projected Tourism Revenue</p>
        </div>
        """, unsafe_allow_html=True)

        if predicted_tourists > 3_000_000:
            st.success("🚀 That's a booming season! High tourist inflow expected.")
        elif predicted_tourists < 500_000:
            st.warning("📉 Expected tourist activity is relatively low. Consider factors like seasonality or restrictions.")
        else:
            st.info("📈 Moderate tourist traffic predicted. Looks like a balanced season.")

# ---------------------------- TAB 2: TRENDS ----------------------------
# ---------------------------- TAB 2: TRENDS ----------------------------
with tab2:
    st.title("📈 Feature Trends Overview")
    st.markdown("Explore how key tourism indicators have changed from 2015 to 2025, specific to each country.")

    selected_country = st.selectbox("Select a country to view trends", ['Bhutan', 'Nepal', 'Thailand', 'India', 'Sri Lanka'])

    # Simulate country-specific trends
    years = np.arange(2015, 2026)

    country_trends = {
        "Bhutan": {
            "Avg Temp (°C)": np.random.normal(18, 1, len(years)),
            "Rainfall (mm)": np.random.normal(130, 15, len(years)),
            "Travel Index": np.linspace(40, 60, len(years)) + np.random.normal(0, 1.5, len(years)),
            "Hotel Occupancy (%)": np.clip(np.random.normal(55, 5, len(years)), 40, 70),
        },
        "Nepal": {
            "Avg Temp (°C)": np.random.normal(22, 1.5, len(years)),
            "Rainfall (mm)": np.random.normal(140, 20, len(years)),
            "Travel Index": np.linspace(50, 70, len(years)) + np.random.normal(0, 2, len(years)),
            "Hotel Occupancy (%)": np.clip(np.random.normal(60, 7, len(years)), 40, 85),
        },
        "Thailand": {
            "Avg Temp (°C)": np.random.normal(30, 1, len(years)),
            "Rainfall (mm)": np.random.normal(200, 25, len(years)),
            "Travel Index": np.linspace(70, 90, len(years)) + np.random.normal(0, 1.5, len(years)),
            "Hotel Occupancy (%)": np.clip(np.random.normal(75, 5, len(years)), 60, 95),
        },
        "India": {
            "Avg Temp (°C)": np.random.normal(28, 1.5, len(years)),
            "Rainfall (mm)": np.random.normal(180, 20, len(years)),
            "Travel Index": np.linspace(60, 85, len(years)) + np.random.normal(0, 2, len(years)),
            "Hotel Occupancy (%)": np.clip(np.random.normal(65, 6, len(years)), 45, 90),
        },
        "Sri Lanka": {
            "Avg Temp (°C)": np.random.normal(27, 1.2, len(years)),
            "Rainfall (mm)": np.random.normal(160, 18, len(years)),
            "Travel Index": np.linspace(55, 75, len(years)) + np.random.normal(0, 2, len(years)),
            "Hotel Occupancy (%)": np.clip(np.random.normal(70, 6, len(years)), 50, 95),
        }
    }

    trends_df = pd.DataFrame({
        "Year": years,
        "Avg Temp (°C)": country_trends[selected_country]["Avg Temp (°C)"],
        "Rainfall (mm)": country_trends[selected_country]["Rainfall (mm)"],
        "Travel Index": country_trends[selected_country]["Travel Index"],
        "Hotel Occupancy (%)": country_trends[selected_country]["Hotel Occupancy (%)"],
    })

    st.line_chart(trends_df.set_index("Year"))
    st.caption(f"📊 Trends shown for **{selected_country}** are synthetically generated.")
