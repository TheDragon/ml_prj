import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import altair as alt



# Load the evaluation CSV
eval_df = pd.read_csv("evaluation_results.csv")

# Compute metrics
mae = mean_absolute_error(eval_df["actual_tourists"], eval_df["predicted_tourists"])
mse = mean_squared_error(eval_df["actual_tourists"], eval_df["predicted_tourists"])
rmse = np.sqrt(mse)
r2 = r2_score(eval_df["actual_tourists"], eval_df["predicted_tourists"])

n = len(eval_df)
p = 14  # number of predictors
adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

# ///////////////////////////////////
# Load your original dataset
df = pd.read_csv("global_tourism_dataset.csv")

# Optional: Create 'year_month' column for smoother timeline plots
df["year_month"] = pd.to_datetime(df["month"] + " " + df["year"].astype(str), format="%b %Y")
# /////////////////////////////////////////

model = joblib.load("tourism_model.pkl")

# Calculate metrics
mae = mean_absolute_error(eval_df["actual_tourists"], eval_df["predicted_tourists"])
rmse = np.sqrt(mean_squared_error(eval_df["actual_tourists"], eval_df["predicted_tourists"]))
r2 = r2_score(eval_df["actual_tourists"], eval_df["predicted_tourists"])

# Calculate residuals
eval_df["residuals"] = eval_df["actual_tourists"] - eval_df["predicted_tourists"]

# Start Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Evaluate", "📈 Trend"])


with tab1:  
    st.title("🌍 Tourist Prediction Dashboard")

    # Year slider + future hint
    st.markdown("🧠 **Note:** Model was trained on data up to 2025. Predictions beyond that are based on trend assumptions.")

    # 📍 Basic Info
    st.header("📍 Basic Information")

    country_region_map = {
        'France': 'Europe',
        'Bhutan': 'Asia',
        'Japan': 'Asia',
        'Thailand': 'Asia',
        'USA': 'Americas'
    }

    country = st.selectbox("Country", list(country_region_map.keys()))
    region = country_region_map[country]
    st.markdown(f"**Region:** {region}")


    month = st.selectbox("Month", ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    year = st.slider("Year", 2010, 2035, 2025)

    # Auto-adjust helper
    def adjust_for_future(base, growth_per_year, year_now=2025):
        if year > year_now:
            return base + (year - year_now) * growth_per_year
        return base

    # 🌦️ Weather & Festival
    with st.expander("🌦️ Weather & Festival Factors"):
        avg_temperature = st.slider("Avg Temperature (°C)", -10.0, 40.0, 22.0)
        rainfall_mm = st.slider("Rainfall (mm)", 0, 500, 80)
        festival = st.selectbox("Festival This Month?", ['Yes', 'No'])

    # 🦠 Restrictions
    with st.expander("🦠 Travel Restrictions & Access"):
        covid = st.selectbox("COVID Restriction Level", ['None', 'Partial', 'Full'])
        visa_required = st.selectbox("Visa Required?", ['Yes', 'No'])

    # 💵 Economic Trends (some values adjusted if future)
    with st.expander("💵 Economic Indicators"):
        fx_rate = st.number_input("Foreign Exchange Rate (1 USD to local)", value=adjust_for_future(65.0, 0.3))
        gdp_growth = st.number_input("GDP Growth Rate (%)", value=adjust_for_future(3.0, 0.1))
        mentions = st.number_input("Internet Mentions", value=adjust_for_future(1000, 50))
        travel_index = st.slider("Global Travel Index", 0.0, 100.0, adjust_for_future(70.0, 0.5))

    # ✈️ Cost & Capacity
    with st.expander("✈️ Travel Cost & Hotel Info"):
        ticket_price = st.slider("Airline Ticket Price (USD)", 100, 2000, int(adjust_for_future(600, 10)))
        occupancy = st.slider("Hotel Occupancy Rate (%)", 0, 100, 75)

    # Log + Cap helper
    def log_cap(val, cap=100):
        return min(np.log1p(val), cap)

    # 🎯 Predict Button
    if st.button("Predict Number of Tourists"):
        input_data = pd.DataFrame([{
            "country": country,
            "region": region,
            "month": month,
            "year": year,
            "festival": 1 if festival == 'Yes' else 0,
            "covid_restriction": {'None': 0, 'Partial': 1, 'Full': 2}[covid],
            "visa_required": 1 if visa_required == 'Yes' else 0,
            "foreign_exchange_rate_log_capped": log_cap(fx_rate),
            "gdp_growth_rate_log_capped": log_cap(gdp_growth),
            "rainfall_mm_log_capped": log_cap(rainfall_mm),
            "internet_mentions_log_capped": log_cap(mentions),
            "global_travel_index_log_capped": log_cap(travel_index),
            "avg_temperature_log_capped": log_cap(avg_temperature),
            "hotel_occupancy_rate_log_capped": log_cap(occupancy),
            "airline_ticket_price_log_capped": log_cap(ticket_price),
            "revenue_usd_log_capped": log_cap(0)  # Not used, but required by model
        }])





        prediction = model.predict(input_data)[0]
        # Show predicted number of tourists
        st.markdown(
            f"""
            <div style='text-align: center; padding: 1rem; background-color: #1e5128; border-radius: 10px; margin-top: 1rem;'>
                <h2 style='color: #ffffff;'>📊 Estimated Number of Tourists</h2>
                <h1 style='color: #7bed9f; font-size: 3rem;'>{int(prediction):,}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )



with tab2:
    st.header("📊 Model Evaluation Summary")

    metrics_data = pd.DataFrame({
        "Metric": [
            "Mean Absolute Error (MAE)",
            "Mean Squared Error (MSE)",
            "Root Mean Squared Error (RMSE)",
            "R-squared (R²)",
            "Adjusted R²"
        ],
        "Value": [
            round(mae, 2),
            round(mse, 2),
            round(rmse, 2),
            round(r2, 4),
            round(adjusted_r2, 4)
        ]
    })

    st.dataframe(metrics_data, use_container_width=True)

with tab3:
    st.header("📈 Tourism Trend Over Time")

    # Filter
    selected_country = st.selectbox("Filter by Country (or view all)", ["All"] + sorted(df["country"].unique().tolist()))

    # Filter data
    trend_df = df if selected_country == "All" else df[df["country"] == selected_country]

    # Tourists over time
    st.subheader("👥 Tourist Trend Over Time")
    tourist_summary = trend_df.groupby("year")["num_tourists"].sum().reset_index()
    tourist_chart = alt.Chart(tourist_summary).mark_line(point=True).encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("num_tourists:Q", title="Total Tourists"),
        tooltip=["year", "num_tourists"]
    ).properties(width=700, height=400)
    st.altair_chart(tourist_chart, use_container_width=True)

