import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk

# Load model
model = joblib.load("random_forest_tourist_model.pkl")

# App Config
st.set_page_config(page_title="Tourist Intelligence Dashboard", layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>🌍 Tourist Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Powered by machine learning insights to forecast tourism trends and performance.</p>", unsafe_allow_html=True)
st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Trends", "📊 Compare"])

# ---------------------------- TAB 1: Predict ----------------------------
with tab1:
    st.markdown("### 🔮 Predict Tourist Volume")
    st.info("Fill out the fields below to predict expected tourist inflow.")

    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox("🌐 Country", ['Bhutan', 'Nepal', 'Thailand', 'India', 'Sri Lanka'])
        region = st.selectbox("📍 Region", ['East', 'West', 'North', 'South', 'Central'])
        month = st.selectbox("🗓️ Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        year = st.slider("📆 Year", 2015, 2025, 2023)

    with col2:
        avg_temperature = st.slider("🌡️ Avg Temperature (°C)", 5.0, 45.0, 25.0)
        rainfall_mm = st.slider("🌧️ Rainfall (mm)", 0.0, 400.0, 100.0)
        festival = st.selectbox("🎉 Festival Season?", [True, False])
        covid_restriction = st.selectbox("🦠 Covid Restrictions?", [True, False])
        visa_required = st.selectbox("🛂 Visa Required?", [True, False])

    gdp_growth_rate = st.slider("📈 GDP Growth Rate (%)", -5.0, 10.0, 3.0)
    foreign_exchange_rate = st.slider("💱 Foreign Exchange Rate", 30.0, 100.0, 70.0)
    internet_mentions = st.slider("🌐 Internet Mentions", 0, 10000, 1000)
    global_travel_index = st.slider("🌍 Global Travel Index", 0.0, 100.0, 50.0)
    airline_ticket_price = st.slider("✈️ Airline Ticket Price ($)", 50.0, 1000.0, 300.0)
    hotel_occupancy_rate = st.slider("🏨 Hotel Occupancy Rate (%)", 0.0, 100.0, 60.0)

    month_num = list(range(1, 13))[["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(month)]

    if st.button("🎯 Predict Now"):
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

        st.success("✅ Prediction Complete")
        st.markdown("---")
        st.markdown(f"""
        <div style='text-align: center;'>
            <h2 style='font-size: 64px;'>🧍 {int(predicted_tourists):,}</h2>
            <p style='font-size: 22px;'>Estimated Tourists in <strong>{country}</strong> during <strong>{month} {year}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        if predicted_tourists > 3_000_000:
            st.success("🚀 High tourist inflow expected!")
        elif predicted_tourists < 500_000:
            st.warning("📉 Tourist activity is relatively low.")
        else:
            st.info("📈 Moderate season. Balanced activity.")

# ---------------------------- TAB 2: Trends ----------------------------
with tab2:
    st.markdown("### 📈 Feature Trends by Country")
    selected_country = st.selectbox("🌍 View trends for", ['Bhutan', 'Nepal', 'Thailand', 'India', 'Sri Lanka'])

    years = np.arange(2015, 2026)
    country_trends = {
        "Bhutan": [18, 130, 50, 55],
        "Nepal": [22, 140, 60, 60],
        "Thailand": [30, 200, 80, 75],
        "India": [28, 180, 70, 65],
        "Sri Lanka": [27, 160, 68, 70]
    }

    base = country_trends[selected_country]
    trends_df = pd.DataFrame({
        "Year": years,
        "Avg Temp (°C)": np.random.normal(base[0], 1, len(years)),
        "Rainfall (mm)": np.random.normal(base[1], 20, len(years)),
        "Travel Index": np.linspace(base[2]-5, base[2]+5, len(years)) + np.random.normal(0, 1.5, len(years)),
        "Hotel Occupancy (%)": np.clip(np.random.normal(base[3], 5, len(years)), 40, 95),
    })

    st.line_chart(trends_df.set_index("Year"))
    st.caption(f"📊 Trends are simulated for {selected_country} and not real data.")

    st.markdown("### 🗺️ Country Location Map")
    coords = {
        "Bhutan": (27.5142, 90.4336),
        "Nepal": (28.3949, 84.1240),
        "Thailand": (15.8700, 100.9925),
        "India": (20.5937, 78.9629),
        "Sri Lanka": (7.8731, 80.7718),
    }
    lat, lon = coords[selected_country]
    map_df = pd.DataFrame({'lat': [lat], 'lon': [lon], 'label': [selected_country]})

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=4),
        layers=[
            pdk.Layer('ScatterplotLayer', data=map_df, get_position='[lon, lat]', get_color='[0, 100, 255, 160]', get_radius=50000),
            pdk.Layer('TextLayer', data=map_df, get_position='[lon, lat]', get_text='label',
                      get_size=16, get_color=[0, 0, 0], get_angle=0, get_alignment_baseline='"bottom"')
        ]
    ))

# ---------------------------- TAB 3: Compare ----------------------------
with tab3:
    st.markdown("### 📊 Compare Countries Side-by-Side")
    compare_countries = st.multiselect("Select countries", ['Bhutan', 'Nepal', 'Thailand', 'India', 'Sri Lanka'], default=['Bhutan', 'Thailand'])

    if compare_countries:
        base_tourism = {"Bhutan": 300_000, "Nepal": 700_000, "Thailand": 4_000_000, "India": 1_000_000, "Sri Lanka": 900_000}
        compare_data = {
            "Country": [],
            "Avg Temp (°C)": [],
            "Rainfall (mm)": [],
            "Travel Index": [],
            "Hotel Occupancy (%)": [],
            "Estimated Tourists": []
        }
        for c in compare_countries:
            compare_data["Country"].append(c)
            compare_data["Avg Temp (°C)"].append(np.random.normal(25 if c != "Bhutan" else 18, 1))
            compare_data["Rainfall (mm)"].append(np.random.normal(150, 20))
            compare_data["Travel Index"].append(np.random.uniform(50, 90))
            compare_data["Hotel Occupancy (%)"].append(np.random.uniform(55, 90))
            compare_data["Estimated Tourists"].append(base_tourism.get(c, 500_000))

        df = pd.DataFrame(compare_data)
        st.dataframe(df.set_index("Country"))

        feature = st.selectbox("📊 Choose feature to visualize", df.columns[1:])
        st.bar_chart(df.set_index("Country")[[feature]])
    else:
        st.info("👈 Select at least one country to compare.")
