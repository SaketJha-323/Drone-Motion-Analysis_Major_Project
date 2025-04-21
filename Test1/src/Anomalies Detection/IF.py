import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Load and preprocess the drone data
def load_and_process_data(file_path):
    data = pd.read_csv(file_path)
    data['X'] /= 1000
    data['Y'] /= 1000
    data['Z'] /= 1000

    data['vx'] = np.gradient(data['X'], data['Time'])
    data['vy'] = np.gradient(data['Y'], data['Time'])
    data['vz'] = np.gradient(data['Z'], data['Time'])

    data['ax'] = np.gradient(data['vx'], data['Time'])
    data['ay'] = np.gradient(data['vy'], data['Time'])
    data['az'] = np.gradient(data['vz'], data['Time'])
    
    return data

# Isolation Forest anomaly detection
def detect_anomalies(data, features):
    clf = IsolationForest(n_estimators=300, max_samples=1.0, contamination="auto", random_state=42)
    scores = clf.fit_predict(data[features])
    return scores

# Plot velocity and acceleration with anomalies
def plot_velocity_acceleration(data, axis, anomaly_indices):
    plt.figure(figsize=(15, 12))

    # Velocity plot
    plt.subplot(2, 1, 1)
    plt.plot(data['Time'], data[f'v{axis}'], label=f'{axis.upper()} Velocity', color='blue', alpha=0.8)
    plt.scatter(data['Time'][anomaly_indices], data[f'v{axis}'][anomaly_indices], color='red', label='Anomalies', edgecolor='black', s=50)
    plt.title(f'{axis.upper()}-Axis Velocity with Anomalies vs Time')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Acceleration plot
    plt.subplot(2, 1, 2)
    plt.plot(data['Time'], data[f'a{axis}'], label=f'{axis.upper()} Acceleration', color='green', alpha=0.8)
    plt.scatter(data['Time'][anomaly_indices], data[f'a{axis}'][anomaly_indices], color='red', label='Anomalies', edgecolor='black', s=50)
    plt.title(f'{axis.upper()}-Axis Acceleration with Anomalies vs Time')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# 2D Isolation Forest scatter plot
def plot_isolation_2d(data, features, x_label, y_label, title):
    clf = IsolationForest(n_estimators=300, max_samples=1.0, contamination="auto", random_state=42)
    clf.fit(data[features])
    scores = clf.predict(data[features])

    inliers = data[features][scores == 1]
    outliers = data[features][scores == -1]

    x_range = np.linspace(data[features[0]].min(), data[features[0]].max(), 100)
    y_range = np.linspace(data[features[1]].min(), data[features[1]].max(), 100)
    xx, yy = np.meshgrid(x_range, y_range)
    zz = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    plt.scatter(inliers[features[0]], inliers[features[1]], c='blue', label='Inliers', s=20)
    plt.scatter(outliers[features[0]], outliers[features[1]], c='red', label='Outliers', s=20)
    plt.contour(xx, yy, zz, levels=[0], linewidths=2, colors='black')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()

# Time-indexed scatter plot
def plot_time_feature_anomalies(data, feature):
    clf = IsolationForest(n_estimators=300, max_samples=1.0, contamination="auto", random_state=42)
    clf.fit(data[[feature]])
    scores = clf.predict(data[[feature]])

    inliers = data[data[feature].index.isin(np.where(scores == 1)[0])]
    outliers = data[data[feature].index.isin(np.where(scores == -1)[0])]

    plt.figure(figsize=(8, 6))
    plt.scatter(inliers.index, inliers[feature], c='blue', label='Inliers', s=20)
    plt.scatter(outliers.index, outliers[feature], c='red', label='Outliers', s=20)
    plt.axhline(0, color='black', linestyle='--', alpha=0.7)
    plt.xlabel('Time Index')
    plt.ylabel(feature)
    plt.title(f'Anomalies in {feature}')
    plt.legend()
    plt.show()

# Generate and print anomaly summary
def generate_anomaly_report(data, features):
    reason_map = {
        'vx': "Sudden speed change along X-axis (maneuver)",
        'vy': "Sudden speed change along Y-axis (maneuver)",
        'vz': "Sudden speed change along Z-axis (altitude jump/fall)",
        'ax': "High acceleration along X-axis (turbulence/control spike)",
        'ay': "High acceleration along Y-axis (side thrust anomaly)",
        'az': "High acceleration along Z-axis (thrust variation)"
    }

    anomaly_summary = []

    for feature in features:
        clf = IsolationForest(n_estimators=300, max_samples=1.0, contamination="auto", random_state=42)
        scores = clf.fit_predict(data[[feature]])
        anomaly_indices = np.where(scores == -1)[0]

        for idx in anomaly_indices:
            anomaly_summary.append({
                'Frame': idx,
                'Time': data['Time'].iloc[idx],
                'Feature': feature,
                'Value': data[feature].iloc[idx],
                'Reason': reason_map.get(feature, "Unknown")
            })

    anomaly_df = pd.DataFrame(anomaly_summary).sort_values(by='Time').reset_index(drop=True)

    print("\n==== Anomaly Detection Report ====\n")
    print(anomaly_df.to_string(index=False))

    return anomaly_df


# Main Execution
if __name__ == "__main__":
    file_path = 'data/processed/updated_drone_data3.csv'
    drone_data = load_and_process_data(file_path)

    # Combined motion features
    motion_features = ['vx', 'vy', 'vz', 'ax', 'ay', 'az']
    motion_anomalies = detect_anomalies(drone_data, motion_features)
    anomaly_indices = np.where(motion_anomalies == -1)[0]

    # Plot velocity and acceleration for X, Y, Z axes
    for axis in ['x', 'y', 'z']:
        plot_velocity_acceleration(drone_data, axis, anomaly_indices)

    # Plot 2D feature isolation
    plot_isolation_2d(drone_data, ['vx', 'vy'], 'Vx (m/s)', 'Vy (m/s)', 'Velocity Anomalies (Vx, Vy)')
    plot_isolation_2d(drone_data, ['ax', 'ay'], 'Ax (m/s²)', 'Ay (m/s²)', 'Acceleration Anomalies (Ax, Ay)')

    # Plot single-feature vs time anomaly detection
    for f in ['vx', 'vy', 'ax', 'ay']:
        plot_time_feature_anomalies(drone_data, f)

    # Print anomaly report
    anomaly_df = generate_anomaly_report(drone_data, motion_features)
