import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Extract the positions of markers and center (C)
marker_positions = [
    ['1_pos_X', '1_pos_Y', '1_pos_Z'],
    ['2_pos_X', '2_pos_Y', '2_pos_Z'],
    ['3_pos_X', '3_pos_Y', '3_pos_Z'],
    ['4_pos_X', '4_pos_Y', '4_pos_Z']
]
center_position = ['C_pos_X', 'C_pos_Y', 'C_pos_Z']

# Function to calculate the angle of rotation between two vectors
def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)

# Store angles
rotation_angles = {f'Marker {i+1}': [] for i in range(4)}

# Visualize using only one frame out of many (e.g., middle one)
frame_index = len(drone_data) // 2

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("📍 Drone 3D Rotation Visualization", fontsize=14)

# Plot center and markers
center = drone_data[center_position].iloc[frame_index].values
ax.scatter(*center, color='black', s=100, label='Center (C)', marker='x')

colors = ['red', 'blue', 'green', 'purple']

for j, marker_pos in enumerate(marker_positions):
    marker = drone_data[marker_pos].iloc[frame_index].values
    vector = marker - center
    angle = calculate_angle(vector, np.array([1, 0, 0]))  # Angle w.r.t X-axis
    rotation_angles[f'Marker {j+1}'].append(angle)

    ax.scatter(*marker, color=colors[j], s=80, label=f'Marker {j+1}')
    ax.quiver(*center, *vector, color=colors[j], length=np.linalg.norm(vector), normalize=True)

    # Annotate rotation degree
    ax.text(*(marker + 0.02), f"{angle:.1f}°", color=colors[j], fontsize=12)

# Set plot limits and labels
ax.set_xlim(center[0] - 0.5, center[0] + 0.5)
ax.set_ylim(center[1] - 0.5, center[1] + 0.5)
ax.set_zlim(center[2] - 0.5, center[2] + 0.5)

ax.set_xlabel('X Axis (m)')
ax.set_ylabel('Y Axis (m)')
ax.set_zlabel('Z Axis (m)')
ax.legend()
plt.tight_layout()
plt.show()

# 📢 Print text output: Average rotation angles over entire flight
print("\n📈 Average Rotation Angles over All Frames:")
for j, marker_pos in enumerate(marker_positions):
    all_angles = []
    for i in range(len(drone_data)):
        center = drone_data[center_position].iloc[i].values
        marker = drone_data[marker_pos].iloc[i].values
        vector = marker - center
        angle = calculate_angle(vector, np.array([1, 0, 0]))
        all_angles.append(angle)
    mean_angle = np.mean(all_angles)
    print(f"🔸 Marker {j+1}: {mean_angle:.2f} degrees")
