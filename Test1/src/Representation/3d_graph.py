import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
file_path = 'Data/Processed/Drone_Data.csv'
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

# Plot settings
x_min, x_max = drone_data[[f'{pos}_X' for pos in positions]].min().min(), drone_data[[f'{pos}_X' for pos in positions]].max().max()
y_min, y_max = drone_data[[f'{pos}_Y' for pos in positions]].min().min(), drone_data[[f'{pos}_Y' for pos in positions]].max().max()
z_min, z_max = drone_data[[f'{pos}_Z' for pos in positions]].min().min(), drone_data[[f'{pos}_Z' for pos in positions]].max().max()

# Plot all data points
def plot_all_data():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set axis labels and limits
    ax.set_title('3D Representation of Drone Movement (All Frames)', fontsize=16)
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_zlabel('Z (meters)', fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Plot trajectories for all markers
    for pos in positions:
        ax.plot(drone_data[f'{pos}_X'], 
                drone_data[f'{pos}_Y'], 
                drone_data[f'{pos}_Z'], 
                label=f'{pos} Trajectory', color=colors[pos], alpha=0.7)

        # Plot starting position
        ax.scatter(drone_data[f'{pos}_X'].iloc[0], 
                   drone_data[f'{pos}_Y'].iloc[0], 
                   drone_data[f'{pos}_Z'].iloc[0], 
                   color=colors[pos], s=50, label=f'{pos} Start', edgecolor='black', marker='o')

        # Plot ending position
        ax.scatter(drone_data[f'{pos}_X'].iloc[-1], 
                   drone_data[f'{pos}_Y'].iloc[-1], 
                   drone_data[f'{pos}_Z'].iloc[-1], 
                   color=colors[pos], s=50, label=f'{pos} End', edgecolor='black', marker='x')

    # Add legend
    ax.legend()

    plt.show()

# Call the function to plot all data
plot_all_data()
