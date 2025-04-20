from enum import auto
from turtle import color
from matplotlib import colors
from matplotlib.pylab import normal
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the calibrated drone data
file_path = 'data/processed/updated_drone_data3.csv'
drone_data = pd.read_csv(file_path)

# Convert position into meters
drone_data['X'] /= 1000
drone_data['Y'] /= 1000
drone_data['Z'] /= 1000

# Calculate time differences
time_diff = np.gradient(drone_data['Time'])

# Calculate velocity from position
drone_data['vx'] = np.gradient(drone_data['X'], drone_data['Time'])
drone_data['vy'] = np.gradient(drone_data['Y'], drone_data['Time'])
drone_data['vz'] = np.gradient(drone_data['Z'], drone_data['Time'])

# Calculate acceleration from velocity
drone_data['ax'] = np.gradient(drone_data['vx'], drone_data['Time'])
drone_data['ay'] = np.gradient(drone_data['vy'], drone_data['Time'])
drone_data['az'] = np.gradient(drone_data['vz'], drone_data['Time'])

# Combine motion data for anomaly detection
motion_data = drone_data[['vx', 'vy', 'vz', 'ax', 'ay', 'az']]

# Apply Isolation Forest for anomaly detection
isolation_forest = IsolationForest(n_estimators=300, max_samples=1.0, contamination="auto", random_state=42)
anomaly_scores = isolation_forest.fit_predict(motion_data)

# Identify anomaly indices
anomaly_indices = np.where(anomaly_scores == -1)[0]

# Identify normal data points
normal_indices = np.where(anomaly_scores == 1)[0]

#for X Axis
# Plot velocity (X, Y, Z) with anomalies
plt.figure(figsize=(15, 12))

# Subplot for X velocity
plt.subplot(2, 1, 1)
plt.plot(drone_data['Time'], drone_data['vx'], label='X Velocity', color='blue', alpha=0.8)
plt.scatter(drone_data['Time'][anomaly_indices], drone_data['vx'][anomaly_indices], color='red', label='Anomalies', edgecolor='black', s=50)
plt.title('X-Axis Velocity with Anomalies vs Time (S)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Subplot for X acceleration
plt.subplot(2, 1, 2)
plt.plot(drone_data['Time'], drone_data['ax'], label='X Acceleration', color='green', alpha=0.8)
plt.scatter(drone_data['Time'][anomaly_indices[:-1]], drone_data['ax'][anomaly_indices[:-1]], color='red', label='Anomalies', edgecolor='black', s=50)
plt.title('X-Axis Acceleration with Anomalies vs Time (S)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
plt.show()


#for Y Axis
# Plot velocity (X, Y, Z) with anomalies
plt.figure(figsize=(15, 12))

# Subplot for Y velocity
plt.subplot(2, 1, 1)
plt.plot(drone_data['Time'], drone_data['vy'], label='X Velocity', color='blue', alpha=0.8)
plt.scatter(drone_data['Time'][anomaly_indices], drone_data['vy'][anomaly_indices], color='red', label='Anomalies', edgecolor='black', s=50)
plt.title('Y-Axis Velocity with Anomalies vs Time (S)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Subplot for Y acceleration
plt.subplot(2, 1, 2)
plt.plot(drone_data['Time'], drone_data['ay'], label='X Acceleration', color='green', alpha=0.8)
plt.scatter(drone_data['Time'][anomaly_indices[:-1]], drone_data['ay'][anomaly_indices[:-1]], color='red', label='Anomalies', edgecolor='black', s=50)
plt.title('Y-AXis Acceleration with Anomalies vs Time (S)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.subplots_adjust(hspace=1)
plt.tight_layout()
plt.show()


#for Z Axis
# Plot velocity (X, Y, Z) with anomalies
plt.figure(figsize=(15, 12))

# Subplot for Z velocity
plt.subplot(2, 1, 1)
plt.plot(drone_data['Time'], drone_data['vz'], label='X Velocity', color='blue', alpha=0.8)
plt.scatter(drone_data['Time'][anomaly_indices], drone_data['vz'][anomaly_indices], color='red', label='Anomalies', edgecolor='black', s=50)
plt.title('Z-Axis Velocity with Anomalies vs Time (S)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Subplot for Z acceleration
plt.subplot(2, 1, 2)
plt.plot(drone_data['Time'], drone_data['az'], label='X Acceleration', color='green', alpha=0.8)
plt.scatter(drone_data['Time'][anomaly_indices[:-1]], drone_data['az'][anomaly_indices[:-1]], color='red', label='Anomalies', edgecolor='black', s=50)
plt.title('Z-Axis Acceleration with Anomalies vs Time (S)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
plt.show()



# velocity and accelration scatter plot
def plot_isolation_forest(features, x_label, y_label, title):
    # Prepare data for Isolation Forest
    feature_data = drone_data[features]
    
    # Train Isolation Forest
    isolation_forest = IsolationForest(n_estimators=300, max_samples=1.0, contamination="auto", random_state=42)
    isolation_forest.fit(feature_data)
    anomaly_scores = isolation_forest.predict(feature_data)

    # Separate inliers and outliers
    inliers = feature_data[anomaly_scores == 1]
    outliers = feature_data[anomaly_scores == -1]

    # Create meshgrid for decision boundary
    x_range = np.linspace(feature_data[features[0]].min(), feature_data[features[0]].max(), 100)
    y_range = np.linspace(feature_data[features[1]].min(), feature_data[features[1]].max(), 100)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    decision_function = isolation_forest.decision_function(grid_points).reshape(xx.shape)

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.scatter(inliers[features[0]], inliers[features[1]], c='blue', label='Inliers', s=20)
    plt.scatter(outliers[features[0]], outliers[features[1]], c='red', label='Outliers', s=20)
    plt.contour(xx, yy, decision_function, levels=[0], linewidths=2, colors='black')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


# Plot for velocity (vx and vy)
plot_isolation_forest(
    features=['vx', 'vy'],
    x_label='Vx (m/s)',
    y_label='Vy (m/s)',
    title='Isolation Forest for Velocity (Vx, Vy)'
)

# Plot for acceleration (ax and ay)
plot_isolation_forest(
    features=['ax', 'ay'],
    x_label='Ax (m/s²)',
    y_label='Ay (m/s²)',
    title='Isolation Forest for Acceleration (Ax, Ay)'
)



# velocity and accelration vs time scatter plot
def plot_anomaly_detection(feature, x_label, title):
    # Prepare data for Isolation Forest
    feature_data = drone_data[[feature]]
    
    # Train Isolation Forest
    isolation_forest = IsolationForest(n_estimators=300, max_samples=1.0, contamination="auto", random_state=42)
    isolation_forest.fit(feature_data)
    anomaly_scores = isolation_forest.predict(feature_data)

    # Separate inliers and outliers
    inliers = feature_data[anomaly_scores == 1]
    outliers = feature_data[anomaly_scores == -1]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(inliers.index, inliers[feature], c='blue', label='Inliers', s=20)
    plt.scatter(outliers.index, outliers[feature], c='red', label='Outliers', s=20)
    plt.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.7)
    plt.xlabel('Time Index')
    plt.ylabel(x_label)
    plt.title(title)
    plt.legend()
    plt.show()

# Graph 1: X-axis velocity
plot_anomaly_detection(
    feature='vx',
    x_label='vx (m/s)',
    title='Isolation Forest for X-axis Velocity (vx)'
)

# Graph 2: Y-axis velocity
plot_anomaly_detection(
    feature='vy',
    x_label='vy (m/s)',
    title='Isolation Forest for Y-axis Velocity (vy)'
)

# Graph 3: X-axis acceleration
plot_anomaly_detection(
    feature='ax',
    x_label='ax (m/s²)',
    title='Isolation Forest for X-axis Acceleration (ax)'
)

# Graph 4: Y-axis acceleration
plot_anomaly_detection(
    feature='ay',
    x_label='ay (m/s²)',
    title='Isolation Forest for Y-axis Acceleration (ay)'
)
