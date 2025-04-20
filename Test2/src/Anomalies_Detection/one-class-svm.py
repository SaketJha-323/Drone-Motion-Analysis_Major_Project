import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
file_path = "data/processed/Modified_Drone_Data.csv"  # Update path if needed
drone_data = pd.read_csv(file_path)

# Ensure the columns are named correctly
drone_data.columns = [
    "Frame","Time",
    "1_pos_X","1_pos_Y","1_pos_Z",
    "2_pos_X","2_pos_Y","2_pos_Z",
    "3_pos_X","3_pos_Y","3_pos_Z",
    "4_pos_X","4_pos_Y","4_pos_Z",
    "C_pos_X","C_pos_Y","C_pos_Z",
]

# Convert position from mm to meters
for col in drone_data.columns[2:]:
    drone_data[col] = drone_data[col] / 100.0

# Extract time
time = drone_data["Time"].values

# Velocity and Acceleration Calculation
def compute_velocity(pos, time):
    return np.diff(pos) / np.diff(time)

def compute_acceleration(vel, time):
    return np.diff(vel) / np.diff(time[1:])

# Compute for each marker
velocities = {}
accelerations = {}
min_length = np.inf

marker_labels = ["1", "2", "3", "4", "C"]
for marker in marker_labels:
    vx = compute_velocity(drone_data[f"{marker}_pos_X"].values, time)
    vy = compute_velocity(drone_data[f"{marker}_pos_Y"].values, time)
    vz = compute_velocity(drone_data[f"{marker}_pos_Z"].values, time)

    ax = compute_acceleration(vx, time)
    ay = compute_acceleration(vy, time)
    az = compute_acceleration(vz, time)

    min_len = min(len(vx), len(vy), len(vz), len(ax), len(ay), len(az))
    min_length = min(min_length, min_len)

    velocities[marker] = np.vstack((vx[:min_len], vy[:min_len], vz[:min_len])).T
    accelerations[marker] = np.vstack((ax[:min_len], ay[:min_len], az[:min_len])).T

# Truncate all to min_length
for marker in marker_labels:
    velocities[marker] = velocities[marker][:min_length]
    accelerations[marker] = accelerations[marker][:min_length]

# Combine all velocity and acceleration data
combined_data = np.hstack([velocities[m] for m in marker_labels] + [accelerations[m] for m in marker_labels])

# Normalize
scaler = StandardScaler()
scaled_data = scaler.fit_transform(combined_data)

# One-Class SVM
svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
preds = svm.fit_predict(scaled_data)
anomaly_indices = np.where(preds == -1)[0]

# Plot Velocity/Acceleration for each marker
# Generate separate graph for each marker
for m in marker_labels:
    v = velocities[m]
    a = accelerations[m]
    
    # Create a figure for each marker
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Plot Velocity for Marker m
    axs[0].plot(v, label=["Vx", "Vy", "Vz"])
    axs[0].scatter(anomaly_indices, v[anomaly_indices, 0], color='red', label="Anomaly Vx", s=15)
    axs[0].set_title(f"Marker {m} Velocity with Anomalies")
    axs[0].legend()
    axs[0].grid(True)

    # Plot Acceleration for Marker m
    axs[1].plot(a, label=["Ax", "Ay", "Az"])
    axs[1].scatter(anomaly_indices[:-1], a[anomaly_indices[:-1], 0], color='red', label="Anomaly Ax", s=15)
    axs[1].set_title(f"Marker {m} Acceleration with Anomalies")
    axs[1].legend()
    axs[1].grid(True)

    # Display the plot for current marker
    plt.tight_layout()
    plt.show()

# # Print Summary Stats for the current marker
# print(f"\n📌 Marker {m} Stats:")
# print(f"  Mean Velocity: {np.mean(v, axis=0)}")
# print(f"  Std Velocity: {np.std(v, axis=0)}")
# print(f"  Mean Acceleration: {np.mean(a, axis=0)}")
# print(f"  Std Acceleration: {np.std(a, axis=0)}")