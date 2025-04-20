import pandas as pd
import numpy as np
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
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Ensure the value is within the valid range for arccos
    return np.degrees(angle)  # Convert from radians to degrees

# Initialize arrays for storing rotation angles (in degrees)
rotation_angles = {f'Marker {i+1}': [] for i in range(4)}

# Iterate over each frame to calculate the rotation angle for each marker relative to the center
for i in range(len(drone_data)):
    center = drone_data[center_position].iloc[i].values  # Position of center C at time i
    for j, marker_pos in enumerate(marker_positions):
        marker = drone_data[marker_pos].iloc[i].values  # Position of marker at time i
        vector_center_to_marker = marker - center
        vector_x_axis = np.array([1, 0, 0])  # Assuming we compare with the X-axis direction
        angle = calculate_angle(vector_center_to_marker, vector_x_axis)
        rotation_angles[f'Marker {j+1}'].append(angle)

# Plot Rotation Angles for Each Marker over Time
plt.figure(figsize=(10, 6))

# Plot each marker's rotation angle over time
for marker in rotation_angles:
    plt.plot(drone_data['Time'], rotation_angles[marker], label=marker, lw=2)

# Set labels, title, and legend for the plot
plt.title("Rotation Angles of Each Marker Over Time", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Rotation Angle (degrees)", fontsize=12)
plt.legend(title="Drone Markers", fontsize=10)
plt.grid(True)

# Add a simple horizontal line to indicate 0 degree (no rotation) position
plt.axhline(0, color='gray', linestyle='--', lw=1)

# Add annotations to make it more kid-friendly
plt.annotate('Starting position', xy=(0, 0), xytext=(1, 20),
             arrowprops=dict(facecolor='blue', shrink=0.05),
             fontsize=12, color='blue')

plt.tight_layout()
plt.show()
