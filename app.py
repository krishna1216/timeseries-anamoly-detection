import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

st.set_page_config(page_title="Time-Series Anomaly Detection", layout="wide")

st.title("📈 LSTM Autoencoder – Time Series Anomaly Detection")
st.write("This app detects anomalies in time-series data using a TensorFlow LSTM Autoencoder.")

# -----------------------------
# Sidebar Parameters
# -----------------------------
st.sidebar.header("⚙️ Configuration")

WINDOW = st.sidebar.slider("Window Size", 20, 100, 50)
EPOCHS = st.sidebar.slider("Training Epochs", 1, 10, 5)
NOISE_LEVEL = st.sidebar.slider("Noise Level", 0.05, 0.5, 0.1)
ANOMALY_COUNT = st.sidebar.slider("Number of Anomalies", 5, 50, 20)

start_btn = st.sidebar.button("🚀 Run Model")

# -----------------------------
# Generate Synthetic Data
# -----------------------------
def generate_data(noise, anomaly_count):
    time = np.arange(0, 2000, 0.1)
    normal_signal = np.sin(0.02 * time) + np.random.normal(0, noise, len(time))

    anomalies = normal_signal.copy()
    anomaly_indices = np.random.choice(len(time), size=anomaly_count, replace=False)
    anomalies[anomaly_indices] += np.random.normal(3, 0.5, anomaly_count)

    return anomalies, anomaly_indices


# -----------------------------
# Create Windowed Data
# -----------------------------
def create_windows(data, window):
    X = []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
    return np.array(X)


# -----------------------------
# Build LSTM Autoencoder
# -----------------------------
def build_model(window):
    model = models.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(32, return_sequences=False),
        layers.RepeatVector(window),
        layers.LSTM(32, return_sequences=True),
        layers.TimeDistributed(layers.Dense(1))
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


if start_btn:

    st.subheader("📊 1. Generated Data")

    data, anomaly_idxs = generate_data(NOISE_LEVEL, ANOMALY_COUNT)

    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(data)
    ax1.scatter(anomaly_idxs, data[anomaly_idxs], color="red")
    st.pyplot(fig1)

    # Prepare windows
    X = create_windows(data, WINDOW)
    X = np.expand_dims(X, axis=-1)

    # Build model
    model = build_model(WINDOW)

    st.subheader("🤖 2. Training LSTM Autoencoder")
    with st.spinner("Training the model..."):
        history = model.fit(
            X, X,
            epochs=EPOCHS,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
    st.success("Training Completed!")

    # Plot training loss
    fig_loss, ax_loss = plt.subplots(figsize=(8,4))
    ax_loss.plot(history.history["loss"], label="Loss")
    ax_loss.plot(history.history["val_loss"], label="Val Loss")
    ax_loss.legend()
    st.pyplot(fig_loss)

    # -----------------------------
    # Reconstruction & Anomaly Detection
    # -----------------------------
    st.subheader("🚨 3. Anomaly Detection")

    X_pred = model.predict(X)
    mse = np.mean(np.power(X - X_pred, 2), axis=(1,2))

    threshold = np.percentile(mse, 95)

    fig_mse, ax_mse = plt.subplots(figsize=(10,4))
    ax_mse.plot(mse, label="Reconstruction Error")
    ax_mse.axhline(threshold, color="red", linestyle="--", label="Threshold")
    ax_mse.legend()
    st.pyplot(fig_mse)

    # Mark detected anomalies
    detected_idxs = np.where(mse > threshold)[0]

    st.subheader("📍 4. Final Anomaly Plot")

    fig_final, ax_final = plt.subplots(figsize=(10,4))
    ax_final.plot(data, label="Time Series")
    ax_final.scatter(detected_idxs, data[detected_idxs], color="red", label="Detected")
    ax_final.legend()
    st.pyplot(fig_final)

    st.success("🎉 Detection Complete!")
else:
    st.info("Set parameters in sidebar and click **Run Model** to start.")
