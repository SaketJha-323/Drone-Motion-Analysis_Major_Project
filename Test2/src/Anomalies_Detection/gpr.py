import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from mpl_toolkits.mplot3d import Axes3D

# Load the calibrated drone data
file_path = "Data/Processed/Modified_Drone_Data.csv"
drone_data = pd.read_csv(file_path)

# Rename columns if needed (example structure for 5 markers)
drone_data.columns = [
    "Frame", "Time",
    "1_pos_X", "1_pos_Y", "1_pos_Z",
    "2_pos_X", "2_pos_Y", "2_pos_Z",
    "3_pos_X", "3_pos_Y", "3_pos_Z",
    "4_pos_X", "4_pos_Y", "4_pos_Z",
    "C_pos_X", "C_pos_Y", "C_pos_Z"
]

# Convert mm to meters
for marker in ["1", "2", "3", "4", "C"]:
    for axis in ["X", "Y", "Z"]:
        drone_data[f"{marker}_pos_{axis}"] /= 1000

# Function to calculate velocity
def calculate_velocity(position, times):
    return np.gradient(position, times)

# GPR and anomaly detection for all 5 markers
marker_ids = ["1", "2", "3", "4", "C"]
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

for marker in marker_ids:
    print(f"\nProcessing Marker {marker}...")

    # Extract time and position
    t = drone_data['Time'].values
    x = drone_data[f"{marker}_pos_X"].values
    y = drone_data[f"{marker}_pos_Y"].values
    z = drone_data[f"{marker}_pos_Z"].values

    # Calculate velocities
    vx = calculate_velocity(x, t)
    vy = calculate_velocity(y, t)
    vz = calculate_velocity(z, t)

    # Training data (just indices)
    X_train = np.arange(len(vx)).reshape(-1, 1)

    # Fit GPR for each velocity component
    gpr_x = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-2).fit(X_train, vx)
    gpr_y = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-2).fit(X_train, vy)
    gpr_z = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-2).fit(X_train, vz)

    pred_x, sigma_x = gpr_x.predict(X_train, return_std=True)
    pred_y, sigma_y = gpr_y.predict(X_train, return_std=True)
    pred_z, sigma_z = gpr_z.predict(X_train, return_std=True)

    # Errors
    error_x = np.abs(pred_x - vx)
    error_y = np.abs(pred_y - vy)
    error_z = np.abs(pred_z - vz)

    # Thresholds (90th percentile)
    threshold_x = np.percentile(error_x, 90)
    threshold_y = np.percentile(error_y, 90)
    threshold_z = np.percentile(error_z, 90)

    # Anomalies
    anomalies_x = error_x > threshold_x
    anomalies_y = error_y > threshold_y
    anomalies_z = error_z > threshold_z
    anomalies = anomalies_x | anomalies_y | anomalies_z

    # Subplots for velocity components (X, Y, Z)
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Velocity X
    axs[0].plot(vx, label="True Velocity (X)", color='blue')
    axs[0].plot(pred_x, label="Predicted Velocity (X)", color='orange')
    axs[0].fill_between(np.arange(len(vx)), pred_x - 2*sigma_x, pred_x + 2*sigma_x, color='orange', alpha=0.2)
    axs[0].scatter(np.where(anomalies_x)[0], vx[anomalies_x], color='red', label='Anomalies')
    axs[0].set_ylabel('Vx (m/s)')
    axs[0].legend()
    axs[0].set_title(f'Marker {marker} - GPR Velocity Components with Anomalies')

    # Velocity Y
    axs[1].plot(vy, label="True Velocity (Y)", color='blue')
    axs[1].plot(pred_y, label="Predicted Velocity (Y)", color='orange')
    axs[1].fill_between(np.arange(len(vy)), pred_y - 2*sigma_y, pred_y + 2*sigma_y, color='orange', alpha=0.2)
    axs[1].scatter(np.where(anomalies_y)[0], vy[anomalies_y], color='red', label='Anomalies')
    axs[1].set_ylabel('Vy (m/s)')
    axs[1].legend()

    # Velocity Z
    axs[2].plot(vz, label="True Velocity (Z)", color='blue')
    axs[2].plot(pred_z, label="Predicted Velocity (Z)", color='orange')
    axs[2].fill_between(np.arange(len(vz)), pred_z - 2*sigma_z, pred_z + 2*sigma_z, color='orange', alpha=0.2)
    axs[2].scatter(np.where(anomalies_z)[0], vz[anomalies_z], color='red', label='Anomalies')
    axs[2].set_ylabel('Vz (m/s)')
    axs[2].set_xlabel('Frame Index')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    # 3D Plot of velocity anomalies
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vx, vy, vz, c='blue', s=5, label='Normal')
    ax.scatter(vx[anomalies], vy[anomalies], vz[anomalies], c='red', s=10, label='Anomaly')
    ax.set_title(f'Marker {marker} - 3D Velocity Anomalies')
    ax.set_xlabel("Vx")
    ax.set_ylabel("Vy")
    ax.set_zlabel("Vz")
    ax.legend()
    plt.show()
