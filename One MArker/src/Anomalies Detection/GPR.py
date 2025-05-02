import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Load the calibrated drone data
file_path = 'data/processed/updated_drone_data2.csv'
drone_data = pd.read_csv(file_path)

# Ensure the columns are named correctly: 'Frame', 'Time', 'X', 'Y', 'Z'
drone_data.columns = ['Frame', 'Time', 'X', 'Y', 'Z']

# Convert data to numpy array for easier manipulation
frames = drone_data['Frame'].values
times = drone_data['Time'].values
x = drone_data['X'].values / 100  # Convert from mm to meters
y = drone_data['Y'].values / 100  # Convert from mm to meters
z = drone_data['Z'].values / 100  # Convert from mm to meters

# Combine X, Y, Z for GPR input
positions = np.vstack((x, y, z)).T

# Function to calculate velocity
def calculate_velocity(position, times):
    return np.diff(position) / np.diff(times)

# Calculate velocities
vx = calculate_velocity(x, times)
vy = calculate_velocity(y, times)
vz = calculate_velocity(z, times)

# Ensure all arrays are the same size by trimming
min_length = min(len(vx), len(vy), len(vz))
vx, vy, vz = vx[:min_length], vy[:min_length], vz[:min_length]

# Combine velocities into a single array for GPR
velocities = np.vstack((vx, vy, vz)).T

# Create training data (input features and outputs)
X_train = np.arange(len(velocities)).reshape(-1, 1)  # Indices as features
y_train = velocities

# Define the kernel for Gaussian Process
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

# Train the Gaussian Process Regressor for each velocity component
gpr_x = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2).fit(X_train, vx)
gpr_y = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2).fit(X_train, vy)
gpr_z = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2).fit(X_train, vz)

# Make predictions and compute the uncertainty
pred_x, sigma_x = gpr_x.predict(X_train, return_std=True)
pred_y, sigma_y = gpr_y.predict(X_train, return_std=True)
pred_z, sigma_z = gpr_z.predict(X_train, return_std=True)

# Calculate the reconstruction error (difference between predicted and actual)
error_x = np.abs(pred_x - vx)
error_y = np.abs(pred_y - vy)
error_z = np.abs(pred_z - vz)

# Define anomaly threshold (e.g., 95th percentile of error)
threshold_x = np.percentile(error_x, 90)
threshold_y = np.percentile(error_y, 90)
threshold_z = np.percentile(error_z, 90)

# Identify anomalies
anomalies_x = error_x > threshold_x
anomalies_y = error_y > threshold_y
anomalies_z = error_z > threshold_z

# Combine anomalies across all components
anomalies = anomalies_x | anomalies_y | anomalies_z

# Visualization of prediction and anomalies
plt.figure(figsize=(12, 6))
plt.plot(vx, label='True Velocity (X)', color='blue')
plt.plot(pred_x, label='Predicted Velocity (X)', color='orange')
plt.fill_between(range(len(sigma_x)), pred_x - 2 * sigma_x, pred_x + 2 * sigma_x, color='orange', alpha=0.2, label='Uncertainty')
plt.scatter(np.where(anomalies_x)[0], vx[anomalies_x], color='red', label='Anomalies (X)')
plt.title('Gaussian Process Regression - Velocity (X)')
plt.legend()
plt.show()

# 3D Visualization of anomalies
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(vx, vy, vz, c='blue', label='Normal Data')
ax.scatter(vx[anomalies], vy[anomalies], vz[anomalies], c='red', label='Anomalies')
ax.set_title('Anomalies in Velocity Data (3D)')
ax.set_xlabel('Vx')
ax.set_ylabel('Vy')
ax.set_zlabel('Vz')
ax.legend()
plt.show()
