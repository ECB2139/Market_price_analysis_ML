import streamlit as st
import pandas as pd
import numpy as np
import os
from prophet import Prophet
import plotly.graph_objs as go
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io

# ---------- CONFIG ----------
DATA_PATH = "Vegetable and Fruits Prices  in India.xlsx"  # your Excel file
SHEET = None  # None => first sheet
MODEL_CACHE_DIR = "models_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

st.set_page_config(page_title="Veg & Fruit Price Forecast", layout="wide")
st.title("🥦 Vegetable & Fruit Price Forecast Dashboard")

# ---------- LOAD DATA ----------
@st.cache_data(ttl=3600)
def load_data(path=DATA_PATH, sheet=SHEET):
    xl = pd.ExcelFile(path)
    sheet_name = sheet if sheet else xl.sheet_names[0]
    df = xl.parse(sheet_name)

    # normalize column names
    df = df.rename(columns={col.strip(): col.strip() for col in df.columns})

    # detect columns automatically
    possible_date_cols = [c for c in df.columns if "date" in c.lower()]
    possible_item_cols = [c for c in df.columns if "item" in c.lower() or "name" in c.lower()]
    possible_price_cols = [c for c in df.columns if "price" in c.lower() or "amount" in c.lower()]

    if not (possible_date_cols and possible_item_cols and possible_price_cols):
        st.error("⚠️ Could not detect Date/Item/Price columns. Please rename properly.")
        st.stop()

    date_col, item_col, price_col = possible_date_cols[0], possible_item_cols[0], possible_price_cols[0]
    df = df[[date_col, item_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, item_col])
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df = df.rename(columns={date_col: "Date", item_col: "Item", price_col: "Price"})
    return df

df = load_data()

# ---------- UI ----------
items = sorted(df['Item'].dropna().unique())
selected_item = st.sidebar.selectbox("Select an item (vegetable/fruit)", items)

freq = st.sidebar.selectbox("Aggregation", ["D", "W", "M"])
freq_map = {"D": "D", "W": "W", "M": "MS"}
agg_freq = freq_map[freq]

horizon_days = st.sidebar.number_input("Forecast horizon (days)", min_value=7, max_value=730, value=90, step=7)
horizon = int(np.ceil(horizon_days / 30)) if freq == "M" else horizon_days

series_df = df[df['Item'] == selected_item].copy()
series_df = series_df.dropna(subset=['Price'])

if len(series_df) < 2:
    st.error("⚠️ Not enough data points for this item.")
    st.stop()

series_df = series_df.set_index('Date').resample(agg_freq)['Price'].mean().reset_index()
series_df = series_df.rename(columns={'Date': 'ds', 'Price': 'y'})

st.sidebar.markdown(f"**Data points:** {len(series_df)}")
st.sidebar.markdown(f"**Date range:** {series_df['ds'].min().date()} → {series_df['ds'].max().date()}")

# ---------- HISTORICAL PLOT ----------
st.subheader(f"📊 Historical Prices — {selected_item}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=series_df['ds'], y=series_df['y'], name='Observed', mode='lines+markers'))
fig.update_layout(height=350, xaxis_title="Date", yaxis_title="Price (₹)")
st.plotly_chart(fig, use_container_width=True)

# ---------- MODEL OPTIONS ----------
st.sidebar.subheader("Model Options")
cap_bool = st.sidebar.checkbox("Use logistic growth (cap limit)", value=False)
cap = st.sidebar.number_input("Cap value", value=float(series_df['y'].max() * 1.2)) if cap_bool else None

train_button = st.sidebar.button("🚀 Train & Forecast")

@st.cache_resource
def fit_prophet_model(df_series, cap=None):
    if cap is not None:
        df_series = df_series.copy()
        df_series['cap'] = cap
        m = Prophet(growth='logistic', yearly_seasonality=True)
    else:
        m = Prophet(yearly_seasonality=True)
    m.fit(df_series)
    return m

# ---------- FORECAST ----------
if train_button:
    with st.spinner("Training model... Please wait ⏳"):
        train_df = series_df.copy()
        if cap is not None:
            train_df['cap'] = cap

        model = fit_prophet_model(train_df, cap)
        model_path = os.path.join(MODEL_CACHE_DIR, f"{selected_item.replace(' ', '_')}_prophet.joblib")
        joblib.dump(model, model_path)
        st.success(f"✅ Model trained and saved for {selected_item}")

        # Predict future
        future = model.make_future_dataframe(periods=horizon, freq=agg_freq)
        if cap is not None:
            future['cap'] = cap
        forecast = model.predict(future)

        # ---------- FULL FORECAST CHART ----------
        st.subheader("📈 Full Forecast")
        full_fig = go.Figure()
        full_fig.add_trace(go.Scatter(x=series_df['ds'], y=series_df['y'], name='Observed', mode='lines'))
        full_fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', mode='lines'))
        full_fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper', mode='lines', line=dict(width=0)))
        full_fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower', mode='lines', line=dict(width=0), fill='tonexty'))
        full_fig.update_layout(height=450, xaxis_title='Date', yaxis_title='Price (₹)')
        st.plotly_chart(full_fig, use_container_width=True)

        # ---------- FUTURE-ONLY TABLE & CHART ----------
        future_df = forecast[forecast['ds'] > series_df['ds'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        st.subheader(f"🔮 Predicted Future Prices — {selected_item}")
        st.dataframe(future_df, use_container_width=True)

        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=future_df['ds'], y=future_df['yhat'], mode='lines+markers', name='Predicted Price'))
        fig_future.update_layout(height=400, xaxis_title="Future Date", yaxis_title="Predicted Price (₹)")
        st.plotly_chart(fig_future, use_container_width=True)

        # ---------- BACKTEST METRICS ----------
        if len(series_df) >= 10:
            split = int(len(series_df) * 0.8)
            train_part = series_df.iloc[:split]
            test_part = series_df.iloc[split:]
            model2 = Prophet(yearly_seasonality=True)
            model2.fit(train_part)
            future_t = model2.make_future_dataframe(periods=len(test_part), freq=agg_freq)
            pred = model2.predict(future_t)
            pred_test = pred.set_index('ds').loc[test_part['ds'], 'yhat'].values

            if len(pred_test) == len(test_part):
                mae = mean_absolute_error(test_part['y'].values, pred_test)
                rmse = np.sqrt(mean_squared_error(test_part['y'].values, pred_test))  # fixed
                st.markdown(f"**Backtest Metrics (last {len(test_part)} points):** MAE = {mae:.2f}, RMSE = {rmse:.2f}")
        else:
            st.info("Not enough points for backtest metrics.")

        # ---------- DOWNLOAD FORECAST ----------
        csv_buf = io.StringIO()
        future_df.to_csv(csv_buf, index=False)
        st.download_button("📥 Download Future Forecast CSV", csv_buf.getvalue(),
                           file_name=f"{selected_item}_future_forecast.csv", mime="text/csv")
else:
    st.info("👈 Choose your item and click **Train & Forecast** to generate predictions.")

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ by Mohammed Aslam J • Prophet + Streamlit")
