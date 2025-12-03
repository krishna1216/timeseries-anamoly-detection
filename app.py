import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Time-Series Anomaly Detection", layout="wide")
st.title("📈 LSTM Autoencoder – Optimized Time Series Anomaly Detection")
st.write("Detect anomalies faster and smoother using a TensorFlow LSTM Autoencoder.")
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
@st.cache_data
def generate_data(noise, anomaly_count):
    time = np.arange(0, 2000, 0.1)
    normal_signal = np.sin(0.02 * time) + np.random.normal(0, noise, len(time))

    anomalies = normal_signal.copy()
    anomaly_indices = np.random.choice(len(time), size=anomaly_count, replace=False)
    anomalies[anomaly_indices] += np.random.normal(3, 0.5, anomaly_count)

    return anomalies, anomaly_indices

# -----------------------------
# Create Windowed Data using tf.data
# -----------------------------
def create_windows_tf(data, window):
    data = data.astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.window(window, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window))
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset

# -----------------------------
# Build LSTM Autoencoder
# -----------------------------
def build_model(window):
    model = models.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(16, return_sequences=False),   # smaller units for faster training
        layers.RepeatVector(window),
        layers.LSTM(16, return_sequences=True),
        layers.TimeDistributed(layers.Dense(1))
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# -----------------------------
# Run Streamlit App
# -----------------------------
if start_btn:

    st.subheader("📊 1. Generated Data")
    data, anomaly_idxs = generate_data(NOISE_LEVEL, ANOMALY_COUNT)

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(data, label="Time Series")
    ax1.scatter(anomaly_idxs, data[anomaly_idxs], color="red", label="Injected Anomalies")
    ax1.set_title("Synthetic Time Series with Anomalies")
    ax1.legend()
    st.pyplot(fig1)
    plt.close(fig1)

    # Prepare windows for model
    X_dataset = create_windows_tf(data, WINDOW)
    X_array = np.array([x for batch in X_dataset for x in batch.numpy()])
    X_array = np.expand_dims(X_array, -1)

    # Build model
    model = build_model(WINDOW)

    st.subheader("🤖 2. Training LSTM Autoencoder")
    with st.spinner("Training the model..."):
        history = model.fit(
            X_array, X_array,
            epochs=EPOCHS,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
    st.success("Training Completed!")

    # Plot training loss
    fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
    ax_loss.plot(history.history["loss"], label="Loss")
    ax_loss.plot(history.history["val_loss"], label="Val Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.legend()
    st.pyplot(fig_loss)
    plt.close(fig_loss)

    # -----------------------------
    # Reconstruction & Anomaly Detection
    # -----------------------------
    st.subheader("🚨 3. Anomaly Detection")
    X_pred = model.predict(X_array)
    mse = np.mean(np.power(X_array - X_pred, 2), axis=(1, 2))

    # Smooth MSE using moving average
    window_ma = max(1, WINDOW // 5)
    mse_smooth = np.convolve(mse, np.ones(window_ma)/window_ma, mode='same')

    threshold = np.percentile(mse_smooth, 95)

    fig_mse, ax_mse = plt.subplots(figsize=(10, 4))
    ax_mse.plot(mse_smooth, label="Smoothed Reconstruction Error")
    ax_mse.axhline(threshold, color="red", linestyle="--", label="Threshold")
    ax_mse.set_title("Smoothed Reconstruction Error with Threshold")
    ax_mse.legend()
    st.pyplot(fig_mse)
    plt.close(fig_mse)

    # Adjust indices
    detected_idxs = np.where(mse_smooth > threshold)[0] + WINDOW // 2

    st.subheader("📍 4. Final Anomaly Plot")
    fig_final, ax_final = plt.subplots(figsize=(10, 4))
    ax_final.plot(data, label="Time Series")
    ax_final.scatter(detected_idxs, data[detected_idxs], color="red", label="Detected Anomalies")
    ax_final.set_title("Detected Anomalies (Smoothed)")
    ax_final.legend()
    st.pyplot(fig_final)
    plt.close(fig_final)

    st.success("🎉 Detection Complete!")

else:
    st.info("Set parameters in sidebar and click **Run Model** to start.")
#finished