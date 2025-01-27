import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load the data
file_path = 'Data/Processed/Modified_Drone_Data.csv'
drone_data = pd.read_csv(file_path)

# Rename columns for convenience
drone_data.columns = ['Frame', 'Time', 
                      '1_pos_X', '1_pos_Y', '1_pos_Z', 
                      '2_pos_X', '2_pos_Y', '2_pos_Z', 
                      '3_pos_X', '3_pos_Y', '3_pos_Z', 
                      '4_pos_X', '4_pos_Y', '4_pos_Z', 
                      'C_pos_X', 'C_pos_Y', 'C_pos_Z']

# Convert positions to meters (from mm)
positions = ['1_pos', '2_pos', '3_pos', '4_pos', 'C_pos']
for pos in positions:
    for axis in ['X', 'Y', 'Z']:
        drone_data[f'{pos}_{axis}'] /= 1000  # Convert mm to meters

# Define colors for each marker
colors = {
    '1_pos': 'blue',
    '2_pos': 'green',
    '3_pos': 'purple',
    '4_pos': 'orange',
    'C_pos': 'red'
}

# Initialize 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Set plot labels and title
ax.set_title('3D Representation of Drone Movement', fontsize=16)
ax.set_xlabel('X (meters)', fontsize=12)
ax.set_ylabel('Y (meters)', fontsize=12)
ax.set_zlabel('Z (meters)', fontsize=12)

# Set axis limits (adjust according to your data range)
x_min, x_max = drone_data[[f'{pos}_X' for pos in positions]].min().min(), drone_data[[f'{pos}_X' for pos in positions]].max().max()
y_min, y_max = drone_data[[f'{pos}_Y' for pos in positions]].min().min(), drone_data[[f'{pos}_Y' for pos in positions]].max().max()
z_min, z_max = drone_data[[f'{pos}_Z' for pos in positions]].min().min(), drone_data[[f'{pos}_Z' for pos in positions]].max().max()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

# Plot trajectories
lines = {}
points = {}
for pos in positions:
    # Add a line and a point for each marker
    lines[pos], = ax.plot([], [], [], label=f'{pos} Trajectory', color=colors[pos], alpha=0.7)
    points[pos], = ax.plot([], [], [], 'o', color=colors[pos], markersize=8, label=f'{pos} Marker')

# Add legend
ax.legend()

# Update function for animation
def update(frame):
    for pos in positions:
        # Update the line data for the trajectory
        lines[pos].set_data(drone_data[f'{pos}_X'][:frame], drone_data[f'{pos}_Y'][:frame])
        lines[pos].set_3d_properties(drone_data[f'{pos}_Z'][:frame])
        
        # Update the current point position
        points[pos].set_data(drone_data[f'{pos}_X'][frame], drone_data[f'{pos}_Y'][frame])
        points[pos].set_3d_properties(drone_data[f'{pos}_Z'][frame])

# Create animation
ani = FuncAnimation(fig, update, frames=len(drone_data), interval=30)  # Adjust interval to match video frame rate (e.g., 30ms for ~30fps)

# Show plot
plt.show()
