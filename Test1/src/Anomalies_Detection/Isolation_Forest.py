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
drone_data["ax1"] = np.gradient(drone_data["1_pos_X"], drone_data["Time"])
drone_data["ax2"] = np.gradient(drone_data["2_pos_X"], drone_data["Time"])
drone_data["ax3"] = np.gradient(drone_data["3_pos_X"], drone_data["Time"])
drone_data["ax4"] = np.gradient(drone_data["4_pos_X"], drone_data["Time"])
drone_data["ax5"] = np.gradient(drone_data["C_pos_X"], drone_data["Time"])

# calculate the acceleration for position Y-Axis
drone_data["ay1"] = np.gradient(drone_data["1_pos_Y"], drone_data["Time"])
drone_data["ay2"] = np.gradient(drone_data["2_pos_Y"], drone_data["Time"])
drone_data["ay3"] = np.gradient(drone_data["3_pos_Y"], drone_data["Time"])
drone_data["ay4"] = np.gradient(drone_data["4_pos_Y"], drone_data["Time"])
drone_data["ay5"] = np.gradient(drone_data["C_pos_Y"], drone_data["Time"])

# calculate the acceleration for the position Z-Axis
drone_data["az1"] = np.gradient(drone_data["1_pos_Z"], drone_data["Time"])
drone_data["az2"] = np.gradient(drone_data["2_pos_Z"], drone_data["Time"])
drone_data["az3"] = np.gradient(drone_data["3_pos_Z"], drone_data["Time"])
drone_data["az4"] = np.gradient(drone_data["4_pos_Z"], drone_data["Time"])
drone_data["az5"] = np.gradient(drone_data["C_pos_Z"], drone_data["Time"])

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

