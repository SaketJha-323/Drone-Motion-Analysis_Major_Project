import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, LSTM, RepeatVector
from keras._tf_keras.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('Data/Processed/Modified_Drone_Data.csv')

# Select only marker position columns
marker_cols = [col for col in df.columns if '_pos_' in col]
data = df[marker_cols].values

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences for LSTM
window_size = 30
sequences = []
for i in range(len(data_scaled) - window_size):
    sequences.append(data_scaled[i:i + window_size])
sequences = np.array(sequences)

# LSTM Autoencoder
timesteps = sequences.shape[1]
features = sequences.shape[2]

inputs = Input(shape=(timesteps, features))
encoded = LSTM(64, activation='relu')(inputs)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(features, activation='sigmoid', return_sequences=True)(decoded)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train model
autoencoder.fit(sequences, sequences, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Reconstruction
reconstructions = autoencoder.predict(sequences)
mse = np.mean(np.mean(np.square(sequences - reconstructions), axis=2), axis=1)

# Anomaly threshold (e.g., 95th percentile)
threshold = np.percentile(mse, 95)
anomalies = mse > threshold

# Plot MSE with anomalies marked
plt.figure(figsize=(12, 6))
plt.plot(mse, label='Reconstruction Error (MSE)')
plt.hlines(threshold, 0, len(mse), colors='r', linestyles='dashed', label=f'Threshold = {threshold:.4f}')
plt.scatter(np.where(anomalies)[0], mse[anomalies], color='red', label='Anomalies')
plt.title("Anomaly Detection using LSTM Autoencoder")
plt.xlabel("Sequence Index")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
