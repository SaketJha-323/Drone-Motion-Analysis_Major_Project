import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv('Data/Processed/Modified_Drone_Data.csv')

# Use one coordinate: e.g., Marker 1's X position
X = df['Time'].values.reshape(-1, 1)
y = df['1_pos_X'].values

# Define kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)

# Fit model
gpr.fit(X, y)

# Predict with uncertainty
y_pred, sigma = gpr.predict(X, return_std=True)

# Residual-based anomaly score
residuals = np.abs(y - y_pred)

# Plot original vs predicted
plt.figure(figsize=(12, 5))
plt.plot(df['Time'], y, label='True Position')
plt.plot(df['Time'], y_pred, label='GPR Prediction', linestyle='--')
plt.fill_between(df['Time'], y_pred - 2*sigma, y_pred + 2*sigma, alpha=0.3, label='95% CI')
plt.title("Gaussian Process Regression on 1_pos_X")
plt.legend()
plt.show()

# Plot anomaly score
plt.plot(df['Time'], residuals)
plt.title("Anomaly Score (Residuals) - GPR")
plt.xlabel("Time")
plt.ylabel("Absolute Error")
plt.show()
