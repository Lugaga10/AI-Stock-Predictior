
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# ==========================================
# 1. App Configuration & Layout
# ==========================================
st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 AI Stock Price Predictor (LSTM)")

# Sidebar for User Inputs
st.sidebar.header("Model Parameters")
TICKER = st.sidebar.text_input("Stock Ticker (e.g., AAPL, GOOG)", "AAPL")
START_DATE = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
END_DATE = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
SEQ_LENGTH = st.sidebar.slider("Lookback Window (Days)", 30, 90, 60)
EPOCHS = st.sidebar.slider("Training Epochs", 1, 50, 25)

# ==========================================
# 2. Helper Functions (Cached for speed)
# ==========================================

@st.cache_data
def fetch_stock_data(ticker, start, end):
    """Fetches data and adds indicators."""
    data = yf.download(ticker, start=start, end=end)
    
    # Handle MultiIndex if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(ticker, level=1, axis=1) if ticker in data.columns.levels[1] else data
        
    if 'Close' not in data.columns:
        return None
        
    df = data[['Close']].copy()
    
    # Add indicators for visualization (not used in this simple model training)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    return df

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ==========================================
# 3. Main App Logic
# ==========================================

if st.sidebar.button("Train Model & Predict"):
    with st.spinner(f"Downloading data for {TICKER}..."):
        df = fetch_stock_data(TICKER, START_DATE, END_DATE)

    if df is None or df.empty:
        st.error("No data found. Please check the Ticker symbol.")
    else:
        # --- Data Preview ---
        st.subheader(f"Raw Data: {TICKER}")
        st.line_chart(df['Close'])

        # --- Preprocessing ---
        data = df[['Close']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - SEQ_LENGTH:]

        X_train, y_train = create_sequences(train_data, SEQ_LENGTH)
        X_test, y_test = create_sequences(test_data, SEQ_LENGTH)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # --- Training ---
        st.subheader("Model Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        model = build_lstm_model((X_train.shape[1], 1))
        
        # Custom Callback to update Streamlit progress bar
        import tensorflow as tf
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / EPOCHS
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {logs['loss']:.4f}")

        with st.spinner("Training LSTM Model... (This may take a minute)"):
            model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, verbose=0, callbacks=[StreamlitCallback()])

        st.success("Training Complete!")

        # --- Predictions ---
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        # --- Metrics ---
        rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
        mae = mean_absolute_error(actual_prices, predictions)

        col1, col2 = st.columns(2)
        col1.metric("Root Mean Squared Error (RMSE)", f"${rmse:.2f}")
        col2.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")

        # --- Visualization ---
        st.subheader("Prediction vs Reality")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(actual_prices, color='black', label='Actual Price')
        ax.plot(predictions, color='green', label='Predicted Price')
        ax.set_title(f'{TICKER} Price Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        st.pyplot(fig)
        
else:
    st.info("Adjust settings in the sidebar and click 'Train Model & Predict' to start.")











