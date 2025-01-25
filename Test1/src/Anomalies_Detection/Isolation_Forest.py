import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the calibrated drone data
file_path = 'Data/Processed/Drone_Data.csv'
drone_data = pd.read_csv(file_path)

# Ensure the columns are named correctly
drone_data.columns = ['Frame', 'Time', 
                      '1_pos_X', '1_pos_Y', '1_pos_Z', 
                      '2_pos_X', '2_pos_Y', '2_pos_Z', 
                      '3_pos_X', '3_pos_Y', '3_pos_Z', 
                      '4_pos_X', '4_pos_Y', '4_pos_Z', 
                      'C_pos_X', 'C_pos_Y', 'C_pos_Z']

# Convert positions to meters (from mm)
positions = ['1_pos', '2_pos', '3_pos', '4_pos', 'C_pos']
data_points = {}
for pos in positions:
    data_points[pos] = {
        'X': drone_data[f'{pos}_X'].values / 1000,
        'Y': drone_data[f'{pos}_Y'].values / 1000,
        'Z': drone_data[f'{pos}_Z'].values / 1000
    }

times = drone_data['Time'].values

# Function to calculate a moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Function to calculate velocity and acceleration
def calculate_velocity(position, time):
    return np.diff(position) / np.diff(time)

def calculate_acceleration(velocity, time):
    return np.diff(velocity) / np.diff(time[1:])

# Process each position
results = {}
window_size = 10
for pos, coords in data_points.items():
    # Apply moving average
    x_avg = moving_average(coords['X'], window_size)
    y_avg = moving_average(coords['Y'], window_size)
    z_avg = moving_average(coords['Z'], window_size)
    times_avg = moving_average(times, window_size)

    # Calculate velocities and accelerations
    vx = calculate_velocity(x_avg, times_avg)
    vy = calculate_velocity(y_avg, times_avg)
    vz = calculate_velocity(z_avg, times_avg)

    ax = calculate_acceleration(vx, times_avg)
    ay = calculate_acceleration(vy, times_avg)
    az = calculate_acceleration(vz, times_avg)

    # Trim all arrays to the same size
    min_length = min(len(vx), len(vy), len(vz), len(ax), len(ay), len(az))
    vx, vy, vz = vx[:min_length], vy[:min_length], vz[:min_length]
    ax, ay, az = ax[:min_length], ay[:min_length], az[:min_length]
    times_trimmed = times_avg[1:len(vx) + 1]

    results[pos] = {
        'times': times_trimmed,
        'vx': vx, 'vy': vy, 'vz': vz,
        'ax': ax, 'ay': ay, 'az': az
    }

# Visualize velocities and accelerations with anomalies
def plot_with_anomalies(data, anomalies, labels, title, ylabel, times):
    plt.figure(figsize=(15, 8))
    for i, (label, color) in enumerate(labels):
        plt.plot(times, data[:, i], label=f'{label}', color=color, alpha=0.7)
        plt.scatter(times[anomalies], data[anomalies, i], color='red', label=f'Anomalies ({label})', edgecolor='black')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Apply Isolation Forest and plot results for each position
for pos, result in results.items():
    # Combine motion data for anomaly detection
    motion_data = np.vstack((result['vx'], result['vy'], result['vz'], result['ax'], result['ay'], result['az'])).T

    # Normalize data using StandardScaler
    scaler = StandardScaler()
    motion_data_scaled = scaler.fit_transform(motion_data)

    # Apply Isolation Forest for anomaly detection
    isolation_forest = IsolationForest(n_estimators=300, max_samples=1.0, contamination=0.1, random_state=42)
    anomaly_scores = isolation_forest.fit_predict(motion_data_scaled)

    # Identify anomaly indices
    anomaly_indices = np.where(anomaly_scores == -1)[0]

    # Plot velocities
    velocity_data = np.vstack((result['vx'], result['vy'], result['vz'])).T
    plot_with_anomalies(velocity_data, anomaly_indices, 
                        [(f'{pos} X Velocity', 'blue'), (f'{pos} Y Velocity', 'green'), (f'{pos} Z Velocity', 'purple')], 
                        f'{pos} Velocities with Anomalies', 'Velocity (m/s)', result['times'])

    # Plot accelerations
    acceleration_data = np.vstack((result['ax'], result['ay'], result['az'])).T
    plot_with_anomalies(acceleration_data, anomaly_indices[:-1], 
                        [(f'{pos} X Acceleration', 'blue'), (f'{pos} Y Acceleration', 'green'), (f'{pos} Z Acceleration', 'purple')], 
                        f'{pos} Accelerations with Anomalies', 'Acceleration (m/s²)', result['times'])

# 3D Visualization of Anomalies
fig = plt.figure(figsize=(10, 8))
ax_3d = fig.add_subplot(111, projection='3d')
for pos, result in results.items():
    motion_data = np.vstack((result['vx'], result['vy'], result['vz'])).T
    anomaly_scores = isolation_forest.fit_predict(scaler.fit_transform(motion_data))
    normal_indices = np.where(anomaly_scores == 1)[0]
    anomaly_indices = np.where(anomaly_scores == -1)[0]

    # Plot normal points
    ax_3d.scatter(motion_data[normal_indices, 0], motion_data[normal_indices, 1], motion_data[normal_indices, 2], 
                  label=f'{pos} Normal', alpha=0.6)

    # Plot anomaly points
    ax_3d.scatter(motion_data[anomaly_indices, 0], motion_data[anomaly_indices, 1], motion_data[anomaly_indices, 2], 
                  label=f'{pos} Anomalies', alpha=0.9)

ax_3d.set_title('3D Visualization of Anomalies', fontsize=14)
ax_3d.set_xlabel('X (m/s)')
ax_3d.set_ylabel('Y (m/s)')
ax_3d.set_zlabel('Z (m/s)')
ax_3d.legend()
plt.show()

# Print statistics for each position
def print_statistics(pos, result):
    dimensions = ['X', 'Y', 'Z']
    for dim, vel, acc in zip(dimensions, [result['vx'], result['vy'], result['vz']], [result['ax'], result['ay'], result['az']]):
        print(f"\n{pos} {dim}-Axis Statistics:")
        print(f"  Average Velocity: {np.mean(vel):.2f} m/s")
        print(f"  Average Acceleration: {np.mean(acc):.2f} m/s²")
        print(f"  Velocity Std Dev: {np.std(vel):.2f} m/s")
        print(f"  Acceleration Std Dev: {np.std(acc):.2f} m/s²")

for pos, result in results.items():
    print_statistics(pos, result)
