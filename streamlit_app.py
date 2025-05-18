import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("random_forest_tourist_model.pkl")

st.set_page_config(page_title="Tourist Prediction Dashboard", layout="centered")

# Create tabs
tab1, tab2 ,tab3 = st.tabs(["🔮 Predict", "📈 Trends", "📊Compare"])

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

        st.markdown("---")
        st.subheader("📊 Prediction Summary")

        # Big number display
        st.markdown(f"""
        <div style='text-align: center; padding: 10px 0;'>
            <h1 style='font-size: 64px; margin-bottom: 0;'>🧍 {int(predicted_tourists):,}</h1>
            <p style='font-size: 20px;'>Estimated Tourists in <strong>{country}</strong> during <strong>{month} {year}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        


        if predicted_tourists > 3_000_000:
            st.success("🚀 That's a booming season! High tourist inflow expected.")
        elif predicted_tourists < 500_000:
            st.warning("📉 Expected tourist activity is relatively low. Consider factors like seasonality or restrictions.")
        else:
            st.info("📈 Moderate tourist traffic predicted. Looks like a balanced season.")
# ---------------------------- TAB 2: TRENDS ----------------------------
with tab2:
    st.title("📈 Feature Trends Overview")
    st.markdown("Explore how key tourism indicators have changed from 2015 to 2025, specific to each country.")

    # Country selection
    selected_country = st.selectbox("🌍 Select a country to view trends", ['Bhutan', 'Nepal', 'Thailand', 'India', 'Sri Lanka'])

    # Simulated feature data by country
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

    # Create DataFrame from the selected country’s trends
    trends_df = pd.DataFrame({
        "Year": years,
        "Avg Temp (°C)": country_trends[selected_country]["Avg Temp (°C)"],
        "Rainfall (mm)": country_trends[selected_country]["Rainfall (mm)"],
        "Travel Index": country_trends[selected_country]["Travel Index"],
        "Hotel Occupancy (%)": country_trends[selected_country]["Hotel Occupancy (%)"],
    })

    st.markdown("#### 📊 Trend Lines")
    st.line_chart(trends_df.set_index("Year"))
    st.caption(f"🧪 Trends for **{selected_country}** are simulated for demo purposes.")

    # ------------------ Country Location Map ------------------
    import pydeck as pdk

    st.markdown("### 🗺️ Country Overview on Map")

    country_coords = {
        "Bhutan": (27.5142, 90.4336),
        "Nepal": (28.3949, 84.1240),
        "Thailand": (15.8700, 100.9925),
        "India": (20.5937, 78.9629),
        "Sri Lanka": (7.8731, 80.7718),
    }

    lat, lon = country_coords.get(selected_country, (20.0, 80.0))

    map_df = pd.DataFrame({
        'lat': [lat],
        'lon': [lon],
        'label': [selected_country]
    })

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=4,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=map_df,
                get_position='[lon, lat]',
                get_color='[0, 100, 255, 160]',
                get_radius=50000,
            ),
            pdk.Layer(
                'TextLayer',
                data=map_df,
                get_position='[lon, lat]',
                get_text='label',
                get_size=16,
                get_color=[0, 0, 0],
                get_angle=0,
                get_alignment_baseline='"bottom"'
            )
        ]
    ))

# ---------------------------- TAB 3: COMPARE COUNTRIES ----------------------------
with tab3:
    st.title("📊 Compare Countries")
    st.markdown("Select multiple countries to compare key tourism indicators for 2025.")

    compare_countries = st.multiselect(
        "Select countries to compare",
        options=['Bhutan', 'Nepal', 'Thailand', 'India', 'Sri Lanka'],
        default=['Bhutan', 'Thailand']
    )

    if compare_countries:
        # Simulate comparison data
        comparison_data = {
            'Country': [],
            'Avg Temp (°C)': [],
            'Rainfall (mm)': [],
            'Travel Index': [],
            'Hotel Occupancy (%)': [],
            'Estimated Tourists': []
        }

        # Simulate values (in real app, load from data source or model)
        for country in compare_countries:
            comparison_data['Country'].append(country)
            comparison_data['Avg Temp (°C)'].append(np.random.normal(25 if country != "Bhutan" else 18, 1))
            comparison_data['Rainfall (mm)'].append(np.random.normal(150, 20))
            comparison_data['Travel Index'].append(np.random.uniform(50, 90))
            comparison_data['Hotel Occupancy (%)'].append(np.random.uniform(55, 90))
            base_tourism = {
                "Bhutan": 300_000,
                "Nepal": 700_000,
                "Thailand": 4_000_000,
                "India": 1_000_000,
                "Sri Lanka": 900_000
            }
            comparison_data['Estimated Tourists'].append(base_tourism.get(country, 500_000))

        compare_df = pd.DataFrame(comparison_data)

        st.markdown("### 📊 Tourism Indicator Comparison (2025)")
        st.dataframe(compare_df.set_index("Country"))

        # Visualize selected features as bar charts
        feature_to_plot = st.selectbox(
            "Choose a feature to visualize",
            options=['Estimated Tourists', 'Avg Temp (°C)', 'Travel Index', 'Hotel Occupancy (%)']
        )

        st.bar_chart(compare_df.set_index("Country")[[feature_to_plot]])
    else:
        st.info("Please select at least one country to compare.")