import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the calibrated drone data
file_path = "Data/Processed/Drone_Data.csv"
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


# convert the position
#  data into meters
drone_data["1_pos_X"] /= 1000
drone_data["1_pos_Y"] /= 1000
drone_data["1_pos_Z"] /= 1000
drone_data["2_pos_X"] /= 1000
drone_data["2_pos_Y"] /= 1000
drone_data["2_pos_Z"] /= 1000
drone_data["3_pos_X"] /= 1000
drone_data["3_pos_Y"] /= 1000
drone_data["3_pos_Z"] /= 1000
drone_data["4_pos_X"] /= 1000
drone_data["4_pos_Y"] /= 1000
drone_data["4_pos_Z"] /= 1000
drone_data["C_pos_X"] /= 1000
drone_data["C_pos_Y"] /= 1000
drone_data["C_pos_Z"] /= 1000

# calculate the time differences
times_diff = np.gradient(drone_data["Time"])

# calculate velocity from position for X-Axis
drone_data["vx1"] = np.gradient(drone_data["1_pos_X"], drone_data["Time"])
drone_data["vx2"] = np.gradient(drone_data["2_pos_X"], drone_data["Time"])
drone_data["vx3"] = np.gradient(drone_data["3_pos_X"], drone_data["Time"])
drone_data["vx4"] = np.gradient(drone_data["4_pos_X"], drone_data["Time"])
drone_data["vx5"] = np.gradient(drone_data["C_pos_X"], drone_data["Time"])

# calucalte the velocity from position for Y-Axis
drone_data["vy1"] = np.gradient(drone_data["1_pos_Y"], drone_data["Time"])
drone_data["vy2"] = np.gradient(drone_data["2_pos_Y"], drone_data["Time"])
drone_data["vy3"] = np.gradient(drone_data["3_pos_Y"], drone_data["Time"])
drone_data["vy4"] = np.gradient(drone_data["4_pos_Y"], drone_data["Time"])
drone_data["vy5"] = np.gradient(drone_data["C_pos_Y"], drone_data["Time"])

# calculate the velocity from position for Z-Axis
drone_data["vz1"] = np.gradient(drone_data["1_pos_Z"], drone_data["Time"])
drone_data["vz2"] = np.gradient(drone_data["2_pos_Z"], drone_data["Time"])
drone_data["vz3"] = np.gradient(drone_data["3_pos_Z"], drone_data["Time"])
drone_data["vz4"] = np.gradient(drone_data["4_pos_Z"], drone_data["Time"])
drone_data["vz5"] = np.gradient(drone_data["C_pos_Z"], drone_data["Time"])

# calculate the acceleration for position X-Axis
drone_data["ax1"] = np.gradient(drone_data["vx1"], drone_data["Time"])
drone_data["ax2"] = np.gradient(drone_data["vx2"], drone_data["Time"])
drone_data["ax3"] = np.gradient(drone_data["vx3"], drone_data["Time"])
drone_data["ax4"] = np.gradient(drone_data["vx4"], drone_data["Time"])
drone_data["ax5"] = np.gradient(drone_data["vx5"], drone_data["Time"])

# calculate the acceleration for position Y-Axis
drone_data["ay1"] = np.gradient(drone_data["vy1"], drone_data["Time"])
drone_data["ay2"] = np.gradient(drone_data["vy2"], drone_data["Time"])
drone_data["ay3"] = np.gradient(drone_data["vy3"], drone_data["Time"])
drone_data["ay4"] = np.gradient(drone_data["vy4"], drone_data["Time"])
drone_data["ay5"] = np.gradient(drone_data["vy5"], drone_data["Time"])

# calculate the acceleration for the position Z-Axis
drone_data["az1"] = np.gradient(drone_data["vz1"], drone_data["Time"])
drone_data["az2"] = np.gradient(drone_data["vz2"], drone_data["Time"])
drone_data["az3"] = np.gradient(drone_data["vz3"], drone_data["Time"])
drone_data["az4"] = np.gradient(drone_data["vz4"], drone_data["Time"])
drone_data["az5"] = np.gradient(drone_data["vz5"], drone_data["Time"])

# combine motion data fro anomalies detection
motion_data = drone_data[
    [
        "vx1","vx2","vx3","vx4","vx5",
        "vy1","vy2","vy3","vy4","vy5",
        "vz1","vz2","vz3","vz4","vz5",
        "ax1","ax2","ax3","ax4","ax5",
        "ay1","ay2","ay3","ay4","ay5",
        "az1","az2","az3","az4","az5",
    ]
]

# Apply Isolation Forest for anomalaies detection
isolation_forest = IsolationForest(n_estimators=300, max_samples=1.0, contamination=0.1, random_state=42)
anomaly_scores = isolation_forest.fit_predict(motion_data)

#identify anomalies indices
anomaly_indices = np.where(anomaly_scores == -1)[0]

#identify non-anomalies indices
normal_indices = np.where(anomaly_scores == 1)[0]

import matplotlib.pyplot as plt
import numpy as np

# Create figure and subplots for X-axis
fig_x, ((ax1_x, ax2_x, ax3_x, ax4_x, ax5_x),
        (ax6_x, ax7_x, ax8_x, ax9_x, ax10_x)) = plt.subplots(2, 5, figsize=(20, 8))
fig_x.suptitle('X-Axis Motion Analysis with Anomaly Detection', fontsize=16)

# Plot X-axis velocities with anomalies
ax1_x.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['vx1'], c='blue', alpha=0.5, s=1, label='Normal')
ax1_x.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['vx1'], c='red', alpha=0.5, s=2, label='Anomaly')
ax1_x.set_title('Marker 1 Velocity')
ax1_x.set_xlabel('Time (s)')
ax1_x.set_ylabel('Velocity (m/s)')
ax1_x.legend()

ax2_x.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['vx2'], c='blue', alpha=0.5, s=1)
ax2_x.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['vx2'], c='red', alpha=0.5, s=2)
ax2_x.set_title('Marker 2 Velocity')
ax2_x.set_xlabel('Time (s)')

ax3_x.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['vx3'], c='blue', alpha=0.5, s=1)
ax3_x.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['vx3'], c='red', alpha=0.5, s=2)
ax3_x.set_title('Marker 3 Velocity')
ax3_x.set_xlabel('Time (s)')

ax4_x.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['vx4'], c='blue', alpha=0.5, s=1)
ax4_x.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['vx4'], c='red', alpha=0.5, s=2)
ax4_x.set_title('Marker 4 Velocity')
ax4_x.set_xlabel('Time (s)')

ax5_x.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['vx5'], c='blue', alpha=0.5, s=1)
ax5_x.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['vx5'], c='red', alpha=0.5, s=2)
ax5_x.set_title('Center Marker Velocity')
ax5_x.set_xlabel('Time (s)')

# Plot X-axis accelerations with anomalies
ax6_x.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['ax1'], c='blue', alpha=0.5, s=1, label='Normal')
ax6_x.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['ax1'], c='red', alpha=0.5, s=2, label='Anomaly')
ax6_x.set_title('Marker 1 Acceleration')
ax6_x.set_xlabel('Time (s)')
ax6_x.set_ylabel('Acceleration (m/s²)')
ax6_x.legend()

ax7_x.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['ax2'], c='blue', alpha=0.5, s=1)
ax7_x.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['ax2'], c='red', alpha=0.5, s=2)
ax7_x.set_title('Marker 2 Acceleration')
ax7_x.set_xlabel('Time (s)')

ax8_x.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['ax3'], c='blue', alpha=0.5, s=1)
ax8_x.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['ax3'], c='red', alpha=0.5, s=2)
ax8_x.set_title('Marker 3 Acceleration')
ax8_x.set_xlabel('Time (s)')

ax9_x.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['ax4'], c='blue', alpha=0.5, s=1)
ax9_x.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['ax4'], c='red', alpha=0.5, s=2)
ax9_x.set_title('Marker 4 Acceleration')
ax9_x.set_xlabel('Time (s)')

ax10_x.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['ax5'], c='blue', alpha=0.5, s=1)
ax10_x.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['ax5'], c='red', alpha=0.5, s=2)
ax10_x.set_title('Center Marker Acceleration')
ax10_x.set_xlabel('Time (s)')

plt.tight_layout()

# Create figure and subplots for Y-axis
fig_y, ((ay1_y, ay2_y, ay3_y, ay4_y, ay5_y),
        (ay6_y, ay7_y, ay8_y, ay9_y, ay10_y)) = plt.subplots(2, 5, figsize=(20, 8))
fig_y.suptitle('Y-Axis Motion Analysis with Anomaly Detection', fontsize=16)

# Plot Y-axis velocities with anomalies
ay1_y.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['vy1'], c='blue', alpha=0.5, s=1, label='Normal')
ay1_y.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['vy1'], c='red', alpha=0.5, s=2, label='Anomaly')
ay1_y.set_title('Marker 1 Velocity')
ay1_y.set_xlabel('Time (s)')
ay1_y.set_ylabel('Velocity (m/s)')
ay1_y.legend()

ay2_y.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['vy2'], c='blue', alpha=0.5, s=1)
ay2_y.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['vy2'], c='red', alpha=0.5, s=2)
ay2_y.set_title('Marker 2 Velocity')
ay2_y.set_xlabel('Time (s)')

ay3_y.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['vy3'], c='blue', alpha=0.5, s=1)
ay3_y.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['vy3'], c='red', alpha=0.5, s=2)
ay3_y.set_title('Marker 3 Velocity')
ay3_y.set_xlabel('Time (s)')

ay4_y.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['vy4'], c='blue', alpha=0.5, s=1)
ay4_y.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['vy4'], c='red', alpha=0.5, s=2)
ay4_y.set_title('Marker 4 Velocity')
ay4_y.set_xlabel('Time (s)')

ay5_y.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['vy5'], c='blue', alpha=0.5, s=1)
ay5_y.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['vy5'], c='red', alpha=0.5, s=2)
ay5_y.set_title('Center Marker Velocity')
ay5_y.set_xlabel('Time (s)')

# Plot Y-axis accelerations with anomalies
ay6_y.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['ay1'], c='blue', alpha=0.5, s=1, label='Normal')
ay6_y.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['ay1'], c='red', alpha=0.5, s=2, label='Anomaly')
ay6_y.set_title('Marker 1 Acceleration')
ay6_y.set_xlabel('Time (s)')
ay6_y.set_ylabel('Acceleration (m/s²)')
ay6_y.legend()

ay7_y.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['ay2'], c='blue', alpha=0.5, s=1)
ay7_y.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['ay2'], c='red', alpha=0.5, s=2)
ay7_y.set_title('Marker 2 Acceleration')
ay7_y.set_xlabel('Time (s)')

ay8_y.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['ay3'], c='blue', alpha=0.5, s=1)
ay8_y.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['ay3'], c='red', alpha=0.5, s=2)
ay8_y.set_title('Marker 3 Acceleration')
ay8_y.set_xlabel('Time (s)')

ay9_y.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['ay4'], c='blue', alpha=0.5, s=1)
ay9_y.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['ay4'], c='red', alpha=0.5, s=2)
ay9_y.set_title('Marker 4 Acceleration')
ay9_y.set_xlabel('Time (s)')

ay10_y.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['ay5'], c='blue', alpha=0.5, s=1)
ay10_y.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['ay5'], c='red', alpha=0.5, s=2)
ay10_y.set_title('Center Marker Acceleration')
ay10_y.set_xlabel('Time (s)')

plt.tight_layout()

# Create figure and subplots for Z-axis
fig_z, ((az1_z, az2_z, az3_z, az4_z, az5_z),
        (az6_z, az7_z, az8_z, az9_z, az10_z)) = plt.subplots(2, 5, figsize=(20, 8))
fig_z.suptitle('Z-Axis Motion Analysis with Anomaly Detection', fontsize=16)

# Plot Z-axis velocities with anomalies
az1_z.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['vz1'], c='blue', alpha=0.5, s=1, label='Normal')
az1_z.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['vz1'], c='red', alpha=0.5, s=2, label='Anomaly')
az1_z.set_title('Marker 1 Velocity')
az1_z.set_xlabel('Time (s)')
az1_z.set_ylabel('Velocity (m/s)')
az1_z.legend()

az2_z.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['vz2'], c='blue', alpha=0.5, s=1)
az2_z.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['vz2'], c='red', alpha=0.5, s=2)
az2_z.set_title('Marker 2 Velocity')
az2_z.set_xlabel('Time (s)')

az3_z.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['vz3'], c='blue', alpha=0.5, s=1)
az3_z.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['vz3'], c='red', alpha=0.5, s=2)
az3_z.set_title('Marker 3 Velocity')
az3_z.set_xlabel('Time (s)')

az4_z.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['vz4'], c='blue', alpha=0.5, s=1)
az4_z.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['vz4'], c='red', alpha=0.5, s=2)
az4_z.set_title('Marker 4 Velocity')
az4_z.set_xlabel('Time (s)')

az5_z.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['vz5'], c='blue', alpha=0.5, s=1)
az5_z.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['vz5'], c='red', alpha=0.5, s=2)
az5_z.set_title('Center Marker Velocity')
az5_z.set_xlabel('Time (s)')

# Plot Z-axis accelerations with anomalies
az6_z.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['az1'], c='blue', alpha=0.5, s=1, label='Normal')
az6_z.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['az1'], c='red', alpha=0.5, s=2, label='Anomaly')
az6_z.set_title('Marker 1 Acceleration')
az6_z.set_xlabel('Time (s)')
az6_z.set_ylabel('Acceleration (m/s²)')
az6_z.legend()

az7_z.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['az2'], c='blue', alpha=0.5, s=1)
az7_z.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['az2'], c='red', alpha=0.5, s=2)
az7_z.set_title('Marker 2 Acceleration')
az7_z.set_xlabel('Time (s)')

az8_z.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['az3'], c='blue', alpha=0.5, s=1)
az8_z.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['az3'], c='red', alpha=0.5, s=2)
az8_z.set_title('Marker 3 Acceleration')
az8_z.set_xlabel('Time (s)')

az9_z.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['az4'], c='blue', alpha=0.5, s=1)
az9_z.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['az4'], c='red', alpha=0.5, s=2)
az9_z.set_title('Marker 4 Acceleration')
az9_z.set_xlabel('Time (s)')

az10_z.scatter(drone_data.iloc[normal_indices]['Time'], drone_data.iloc[normal_indices]['az5'], c='blue', alpha=0.5, s=1)
az10_z.scatter(drone_data.iloc[anomaly_indices]['Time'], drone_data.iloc[anomaly_indices]['az5'], c='red', alpha=0.5, s=2)
az10_z.set_title('Center Marker Acceleration')
az10_z.set_xlabel('Time (s)')

plt.tight_layout()
plt.show()
