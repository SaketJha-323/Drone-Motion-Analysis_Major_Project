import numpy as np
from scipy.stats import zscore
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Load the data
file_path = 'Data/Processed/Modified_Drone_Data.csv'
# Create dummy data if file doesn't exist (for testing purposes)
if not os.path.exists(file_path):
    print(f"Warning: File not found at {file_path}. Creating dummy data.")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data = {
        'Frame': range(100),
        'Time': [i * 0.1 for i in range(100)],
        'Mocap_1_pos_X': [1000 + i * 10 for i in range(100)], 'Mocap_1_pos_Y': [1000 + i * 5 for i in range(100)], 'Mocap_1_pos_Z': [500 + (i % 20) * 10 for i in range(100)],
        'Mocap_2_pos_X': [1100 + i * 10 for i in range(100)], 'Mocap_2_pos_Y': [900 + i * 5 for i in range(100)], 'Mocap_2_pos_Z': [510 + (i % 20) * 10 for i in range(100)],
        'Mocap_3_pos_X': [900 + i * 10 for i in range(100)], 'Mocap_3_pos_Y': [900 + i * 5 for i in range(100)], 'Mocap_3_pos_Z': [490 + (i % 20) * 10 for i in range(100)],
        'Mocap_4_pos_X': [1000 + i * 10 for i in range(100)], 'Mocap_4_pos_Y': [1100 + i * 5 for i in range(100)], 'Mocap_4_pos_Z': [500 + (i % 20) * 10 for i in range(100)],
        'Mocap_C_pos_X': [1000 + i * 10 for i in range(100)], 'Mocap_C_pos_Y': [1000 + i * 5 for i in range(100)], 'Mocap_C_pos_Z': [500 + (i % 20) * 10 for i in range(100)],
    }
    drone_data = pd.DataFrame(data)
    drone_data.to_csv(file_path, index=False)
else:
    drone_data = pd.read_csv(file_path)

# Rename columns for convenience (adjust if original names differ slightly)
# Assuming original names might be like 'Mocap_1_pos_X' etc. based on typical datasets
# If your names are exactly as in the original script ('1_pos_X'), adjust the renaming logic or skip if not needed.
rename_map = {
    'Frame': 'Frame', 'Time': 'Time',
    'Mocap_1_pos_X': '1_pos_X', 'Mocap_1_pos_Y': '1_pos_Y', 'Mocap_1_pos_Z': '1_pos_Z',
    'Mocap_2_pos_X': '2_pos_X', 'Mocap_2_pos_Y': '2_pos_Y', 'Mocap_2_pos_Z': '2_pos_Z',
    'Mocap_3_pos_X': '3_pos_X', 'Mocap_3_pos_Y': '3_pos_Y', 'Mocap_3_pos_Z': '3_pos_Z',
    'Mocap_4_pos_X': '4_pos_X', 'Mocap_4_pos_Y': '4_pos_Y', 'Mocap_4_pos_Z': '4_pos_Z',
    'Mocap_C_pos_X': 'C_pos_X', 'Mocap_C_pos_Y': 'C_pos_Y', 'Mocap_C_pos_Z': 'C_pos_Z'
}
# Apply renaming only if the old names exist
cols_to_rename = {k: v for k, v in rename_map.items() if k in drone_data.columns}
drone_data.rename(columns=cols_to_rename, inplace=True)

# Define marker identifiers
positions = ['1_pos', '2_pos', '3_pos', '4_pos', 'C_pos']

# Convert positions to meters (from mm)
for pos in positions:
    for axis in ['X', 'Y', 'Z']:
        col_name = f'{pos}_{axis}'
        if col_name in drone_data.columns:
            drone_data[col_name] /= 1000  # Convert mm to meters
        else:
            print(f"Warning: Column {col_name} not found in DataFrame.")


# Define colors for each marker
colors = {
    '1_pos': 'blue',
    '2_pos': 'green',
    '3_pos': 'purple',
    '4_pos': 'orange',
    'C_pos': 'red'
}

# Calculate overall axis limits for consistent scaling across plots
all_pos_cols = [f'{pos}_{axis}' for pos in positions for axis in ['X', 'Y', 'Z'] if f'{pos}_{axis}' in drone_data.columns]
if not all_pos_cols:
    raise ValueError("No position columns found after renaming and checking.")

x_cols = [col for col in all_pos_cols if '_X' in col]
y_cols = [col for col in all_pos_cols if '_Y' in col]
z_cols = [col for col in all_pos_cols if '_Z' in col]

x_min, x_max = drone_data[x_cols].min().min(), drone_data[x_cols].max().max()
y_min, y_max = drone_data[y_cols].min().min(), drone_data[y_cols].max().max()
z_min, z_max = drone_data[z_cols].min().min(), drone_data[z_cols].max().max()

# Add some padding to the limits
padding_factor = 0.1
x_range = x_max - x_min
y_range = y_max - y_min
z_range = z_max - z_min
x_min -= x_range * padding_factor
x_max += x_range * padding_factor
y_min -= y_range * padding_factor
y_max += y_range * padding_factor
z_min -= z_range * padding_factor
z_max += z_range * padding_factor


def plot_single_marker_3d(data, marker_id, color, xlim, ylim, zlim):
    """Plots the 3D trajectory for a single marker."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    x_col, y_col, z_col = f'{marker_id}_X', f'{marker_id}_Y', f'{marker_id}_Z'

    if not all(col in data.columns for col in [x_col, y_col, z_col]):
        print(f"Skipping plot for {marker_id}: Columns not found.")
        plt.close(fig) # Close the empty figure
        return

    # Plot trajectory
    ax.plot(data[x_col], data[y_col], data[z_col],
            label=f'{marker_id} Trajectory', color=color, alpha=0.8)

    # Plot starting position
    ax.scatter(data[x_col].iloc[0], data[y_col].iloc[0], data[z_col].iloc[0],
               color=color, s=80, label='Start', edgecolor='black', marker='o', depthshade=False)

    # Plot ending position
    ax.scatter(data[x_col].iloc[-1], data[y_col].iloc[-1], data[z_col].iloc[-1],
               color=color, s=80, label='End', edgecolor='black', marker='X', depthshade=False)

    # Set axis labels, title, and limits
    ax.set_title(f'3D Trajectory for Marker {marker_id.split("_")[0]}', fontsize=16)
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_zlabel('Z (meters)', fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # Improve layout and add legend
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

# --- Generate Plots (Approach 1) ---
print("Generating separate 3D plots for each marker...")
for pos in positions:
    plot_single_marker_3d(drone_data, pos, colors[pos], (x_min, x_max), (y_min, y_max), (z_min, z_max))

# Choose marker to analyze anomalies (e.g., 'C_pos' as central drone point)
marker = 'C_pos'

# Compute velocities and accelerations for X, Y, Z
for axis in ['X', 'Y', 'Z']:
    pos_col = f'{marker}_{axis}'
    vel_col = f'{marker}_V_{axis}'
    acc_col = f'{marker}_A_{axis}'

    # Velocity: first derivative of position w.r.t. time
    drone_data[vel_col] = np.gradient(drone_data[pos_col], drone_data['Time'])

    # Acceleration: first derivative of velocity w.r.t. time
    drone_data[acc_col] = np.gradient(drone_data[vel_col], drone_data['Time'])

# Function to detect anomalies using z-score
def detect_anomalies(series, threshold=3):
    z_scores = zscore(series)
    return np.abs(z_scores) > threshold

# Function to plot Frame vs value (Velocity/Acceleration) with anomalies
def plot_anomaly_graph(data, col1, col2, label, anomaly_label):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['Time'], data[col1], label=label, color='blue')
    anomalies = detect_anomalies(data[col1])
    ax.scatter(data['Time'][anomalies], data[col1][anomalies], color='red', label=anomaly_label, zorder=5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(label)
    ax.set_title(f"{label} Anomalies for {col2}")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Axis Pairs for Velocity and Acceleration
axis_pairs = [('X', 'Y'), ('Y', 'Z'), ('X', 'Z')]

for a1, a2 in axis_pairs:
    vel_label = f"{marker}_V_{a1}{a2}"
    acc_label = f"{marker}_A_{a1}{a2}"

    # Combine velocities and accelerations by computing magnitude across axis pair
    drone_data[vel_label] = np.sqrt(drone_data[f'{marker}_V_{a1}']**2 + drone_data[f'{marker}_V_{a2}']**2)
    drone_data[acc_label] = np.sqrt(drone_data[f'{marker}_A_{a1}']**2 + drone_data[f'{marker}_A_{a2}']**2)

    # Plot Frame vs Velocity magnitude
    plot_anomaly_graph(drone_data, vel_label, f"{a1}-{a2}", f"Velocity ({a1}-{a2})", "Velocity Anomaly")

    # Plot Frame vs Acceleration magnitude
    plot_anomaly_graph(drone_data, acc_label, f"{a1}-{a2}", f"Acceleration ({a1}-{a2})", "Acceleration Anomaly")

# Now for individual X, Y, Z
for axis in ['X', 'Y', 'Z']:
    vel_col = f'{marker}_V_{axis}'
    acc_col = f'{marker}_A_{axis}'

    # Plot Frame vs Velocity
    plot_anomaly_graph(drone_data, vel_col, axis, f"Velocity ({axis})", "Velocity Anomaly")

    # Plot Frame vs Acceleration
    plot_anomaly_graph(drone_data, acc_col, axis, f"Acceleration ({axis})", "Acceleration Anomaly")
