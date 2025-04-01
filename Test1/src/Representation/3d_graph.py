# import pandas as pd
# import numpy as np # Import numpy for NaN handling
# import matplotlib.pyplot as plt
# import os

# # --- Configuration & Data Loading ---
# file_path = 'Data/Processed/Modified_Drone_Data.csv'
# # output_dir = 'Plots/Kinematics' # Directory to save plots (optional)
# os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist

# # Create dummy data if file doesn't exist
# if not os.path.exists(file_path):
#     print(f"Warning: File not found at {file_path}. Creating dummy data.")
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     n_frames = 200
#     time_step = 0.02 # seconds (e.g., 50 Hz)
#     data = {
#         'Frame': range(n_frames),
#         'Time': [i * time_step for i in range(n_frames)],
#         # Simulate some movement with noise
#         'Mocap_1_pos_X': 1000 + 500 * np.sin(np.linspace(0, 4*np.pi, n_frames)) + np.random.normal(0, 10, n_frames),
#         'Mocap_1_pos_Y': 1500 + 400 * np.cos(np.linspace(0, 4*np.pi, n_frames)) + np.random.normal(0, 10, n_frames),
#         'Mocap_1_pos_Z': 500 + 100 * np.sin(np.linspace(0, 2*np.pi, n_frames)) + np.random.normal(0, 5, n_frames),
#         'Mocap_2_pos_X': 1100 + 500 * np.sin(np.linspace(0, 4*np.pi, n_frames)) + np.random.normal(0, 10, n_frames),
#         'Mocap_2_pos_Y': 1400 + 400 * np.cos(np.linspace(0, 4*np.pi, n_frames)) + np.random.normal(0, 10, n_frames),
#         'Mocap_2_pos_Z': 510 + 100 * np.sin(np.linspace(0, 2*np.pi, n_frames)) + np.random.normal(0, 5, n_frames),
#         'Mocap_3_pos_X': 900 + 500 * np.sin(np.linspace(0, 4*np.pi, n_frames)) + np.random.normal(0, 10, n_frames),
#         'Mocap_3_pos_Y': 1400 + 400 * np.cos(np.linspace(0, 4*np.pi, n_frames)) + np.random.normal(0, 10, n_frames),
#         'Mocap_3_pos_Z': 490 + 100 * np.sin(np.linspace(0, 2*np.pi, n_frames)) + np.random.normal(0, 5, n_frames),
#         'Mocap_4_pos_X': 1000 + 500 * np.sin(np.linspace(0, 4*np.pi, n_frames)) + np.random.normal(0, 10, n_frames),
#         'Mocap_4_pos_Y': 1600 + 400 * np.cos(np.linspace(0, 4*np.pi, n_frames)) + np.random.normal(0, 10, n_frames),
#         'Mocap_4_pos_Z': 500 + 100 * np.sin(np.linspace(0, 2*np.pi, n_frames)) + np.random.normal(0, 5, n_frames),
#         'Mocap_C_pos_X': 1000 + 500 * np.sin(np.linspace(0, 4*np.pi, n_frames)) + np.random.normal(0, 10, n_frames),
#         'Mocap_C_pos_Y': 1500 + 400 * np.cos(np.linspace(0, 4*np.pi, n_frames)) + np.random.normal(0, 10, n_frames),
#         'Mocap_C_pos_Z': 500 + 100 * np.sin(np.linspace(0, 2*np.pi, n_frames)) + np.random.normal(0, 5, n_frames),
#     }
#     drone_data = pd.DataFrame(data)
#     drone_data.to_csv(file_path, index=False)
# else:
#     drone_data = pd.read_csv(file_path)

# # --- Data Preprocessing ---

# # Rename columns for convenience (adjust map if original names differ)
# rename_map = {
#     'Frame': 'Frame', 'Time': 'Time',
#     'Mocap_1_pos_X': '1_pos_X', 'Mocap_1_pos_Y': '1_pos_Y', 'Mocap_1_pos_Z': '1_pos_Z',
#     'Mocap_2_pos_X': '2_pos_X', 'Mocap_2_pos_Y': '2_pos_Y', 'Mocap_2_pos_Z': '2_pos_Z',
#     'Mocap_3_pos_X': '3_pos_X', 'Mocap_3_pos_Y': '3_pos_Y', 'Mocap_3_pos_Z': '3_pos_Z',
#     'Mocap_4_pos_X': '4_pos_X', 'Mocap_4_pos_Y': '4_pos_Y', 'Mocap_4_pos_Z': '4_pos_Z',
#     'Mocap_C_pos_X': 'C_pos_X', 'Mocap_C_pos_Y': 'C_pos_Y', 'Mocap_C_pos_Z': 'C_pos_Z'
# }
# cols_to_rename = {k: v for k, v in rename_map.items() if k in drone_data.columns}
# drone_data.rename(columns=cols_to_rename, inplace=True)

# # Define marker identifiers and check if columns exist
# positions = ['1_pos', '2_pos', '3_pos', '4_pos', 'C_pos']
# axes = ['X', 'Y', 'Z']
# pos_cols_exist = True
# for pos in positions:
#     for axis in axes:
#         col_name = f'{pos}_{axis}'
#         if col_name not in drone_data.columns:
#             print(f"Warning: Position column {col_name} not found.")
#             pos_cols_exist = False
# if not pos_cols_exist:
#     raise ValueError("Essential position columns are missing. Cannot proceed.")

# # Convert positions to meters (from mm)
# for pos in positions:
#     for axis in axes:
#         col_name = f'{pos}_{axis}'
#         drone_data[col_name] /= 1000.0  # Convert mm to meters

# # --- Calculate Velocity and Acceleration ---

# def calculate_kinematics(df, markers, axes):
#     """Calculates velocity and acceleration for specified markers and axes."""
#     df_kin = df.copy()
#     if 'Time' not in df_kin.columns:
#         raise KeyError("DataFrame must contain a 'Time' column.")

#     # Ensure time is sorted
#     df_kin = df_kin.sort_values('Time').reset_index(drop=True)

#     # Calculate delta_t (time difference between consecutive frames)
#     delta_t = df_kin['Time'].diff() # Series of time steps

#     # Calculate Velocity (m/s)
#     for marker in markers:
#         for axis in axes:
#             pos_col = f'{marker}_{axis}'
#             vel_col = f'{marker}_vel_{axis}'
#             if pos_col in df_kin.columns:
#                 # Change in position / change in time
#                 df_kin[vel_col] = df_kin[pos_col].diff() / delta_t
#             else:
#                 print(f"Skipping velocity calculation for missing column: {pos_col}")

#     # Calculate Acceleration (m/s^2)
#     # Note: Acceleration calculation uses the same delta_t as velocity
#     for marker in markers:
#         for axis in axes:
#             vel_col = f'{marker}_vel_{axis}'
#             acc_col = f'{marker}_acc_{axis}'
#             if vel_col in df_kin.columns:
#                  # Change in velocity / change in time
#                  # We reuse the delta_t calculated earlier. It aligns correctly
#                  # because acc[i] = (vel[i] - vel[i-1]) / (time[i] - time[i-1])
#                 df_kin[acc_col] = df_kin[vel_col].diff() / delta_t
#             else:
#                  print(f"Skipping acceleration calculation for missing column: {vel_col}")

#     # First row of velocity and first two rows of acceleration will be NaN due to diff()
#     # You might want to dropna based on a key acceleration column or handle otherwise
#     # For plotting, we can often just let matplotlib handle NaNs (it won't connect gaps)
#     # Or drop rows where acceleration is NaN:
#     # key_acc_col = f"{markers[0]}_acc_{axes[0]}" # Example key column
#     # if key_acc_col in df_kin.columns:
#     #      df_kin = df_kin.dropna(subset=[key_acc_col]).reset_index(drop=True)

#     return df_kin

# # Calculate kinematics
# print("Calculating velocity and acceleration...")
# drone_kinematics = calculate_kinematics(drone_data, positions, axes)
# print("Calculation complete.")
# # print(drone_kinematics.head()) # Optional: inspect the first few rows


# # --- Plotting ---

# # Define colors for markers (consistent with previous script if needed)
# colors = {
#     '1_pos': 'blue', '2_pos': 'green', '3_pos': 'purple',
#     '4_pos': 'orange', 'C_pos': 'red'
# }
# # Define line styles or colors for components if needed (e.g., X=solid, Y=dashed, Z=dotted)
# component_colors = {'X': 'tab:blue', 'Y': 'tab:red', 'Z': 'tab:green'}
# component_styles = {'X': '-', 'Y': '--', 'Z': ':'}

# # Find overall min/max for velocity and acceleration for consistent y-axis limits
# vel_cols = [f'{m}_vel_{ax}' for m in positions for ax in axes if f'{m}_vel_{ax}' in drone_kinematics.columns]
# acc_cols = [f'{m}_acc_{ax}' for m in positions for ax in axes if f'{m}_acc_{ax}' in drone_kinematics.columns]

# # Calculate limits, ignoring NaN values
# vel_min = drone_kinematics[vel_cols].min().min() if vel_cols else 0
# vel_max = drone_kinematics[vel_cols].max().max() if vel_cols else 1
# acc_min = drone_kinematics[acc_cols].min().min() if acc_cols else 0
# acc_max = drone_kinematics[acc_cols].max().max() if acc_cols else 1

# # Add some padding to limits
# vel_range = vel_max - vel_min
# acc_range = acc_max - acc_min
# vel_min -= vel_range * 0.1
# vel_max += vel_range * 0.1
# acc_min -= acc_range * 0.1
# acc_max += acc_range * 0.1
# # Ensure range is not zero if data is flat
# if vel_max == vel_min: vel_max += 0.5; vel_min -= 0.5
# if acc_max == acc_min: acc_max += 0.5; acc_min -= 0.5


# def plot_marker_kinematics(df, marker_id, time_col='Time', axes=['X', 'Y', 'Z'],
#                            comp_colors=None, comp_styles=None,
#                            vel_ylim=None, acc_ylim=None, save_dir=None):
#     """
#     Plots velocity and acceleration components vs. time for a single marker.
#     Creates two separate plots: one for velocity, one for acceleration.
#     """
#     marker_label = marker_id.split('_')[0] # Get '1', '2', 'C' etc.
#     fig_vel, ax_vel = plt.subplots(figsize=(12, 5))
#     fig_acc, ax_acc = plt.subplots(figsize=(12, 5))

#     plot_occurred_vel = False
#     plot_occurred_acc = False

#     for axis in axes:
#         vel_col = f'{marker_id}_vel_{axis}'
#         acc_col = f'{marker_id}_acc_{axis}'
#         color = comp_colors.get(axis, 'black') if comp_colors else 'black'
#         style = comp_styles.get(axis, '-') if comp_styles else '-'

#         # Plot Velocity
#         if vel_col in df.columns:
#             ax_vel.plot(df[time_col], df[vel_col],
#                         label=f'Vel {axis}', color=color, linestyle=style, alpha=0.8)
#             plot_occurred_vel = True
#         else:
#             print(f"Velocity column {vel_col} not found for plotting.")

#         # Plot Acceleration
#         if acc_col in df.columns:
#             ax_acc.plot(df[time_col], df[acc_col],
#                         label=f'Acc {axis}', color=color, linestyle=style, alpha=0.8)
#             plot_occurred_acc = True
#         else:
#             print(f"Acceleration column {acc_col} not found for plotting.")

#     # --- Finalize Velocity Plot ---
#     if plot_occurred_vel:
#         ax_vel.set_title(f'Marker {marker_label} - Velocity Components vs. Time', fontsize=16)
#         ax_vel.set_xlabel('Time (seconds)', fontsize=12)
#         ax_vel.set_ylabel('Velocity (m/s)', fontsize=12)
#         if vel_ylim:
#             ax_vel.set_ylim(vel_ylim)
#         ax_vel.legend(fontsize=10)
#         ax_vel.grid(True, linestyle='--', alpha=0.6)
#         fig_vel.tight_layout()
#         if save_dir:
#              fig_vel.savefig(os.path.join(save_dir, f'marker_{marker_label}_velocity.png'))
#         plt.show(fig_vel)
#     else:
#         plt.close(fig_vel) # Close empty figure

#     # --- Finalize Acceleration Plot ---
#     if plot_occurred_acc:
#         ax_acc.set_title(f'Marker {marker_label} - Acceleration Components vs. Time', fontsize=16)
#         ax_acc.set_xlabel('Time (seconds)', fontsize=12)
#         ax_acc.set_ylabel('Acceleration (m/s²)', fontsize=12)
#         if acc_ylim:
#             ax_acc.set_ylim(acc_ylim)
#         ax_acc.legend(fontsize=10)
#         ax_acc.grid(True, linestyle='--', alpha=0.6)
#         fig_acc.tight_layout()
#         if save_dir:
#             fig_acc.savefig(os.path.join(save_dir, f'marker_{marker_label}_acceleration.png'))
#         plt.show(fig_acc)
#     else:
#         plt.close(fig_acc) # Close empty figure


# # --- Generate Plots ---
# print("\nGenerating kinematics plots for each marker...")
# for pos_id in positions: # e.g., '1_pos', '2_pos', ...
#     plot_marker_kinematics(drone_kinematics, pos_id, axes=axes,
#                            comp_colors=component_colors, comp_styles=component_styles,
#                            vel_ylim=(vel_min, vel_max), acc_ylim=(acc_min, acc_max),
#                            save_dir=output_dir) # Pass the save directory

# print(f"\nPlots saved to {output_dir} (if save_dir was specified)")



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

# ============================================================================

# Approach 2: 2D Projections (X-Y, X-Z, Y-Z) for Each Marker

# This shows the path projected onto the standard 2D planes. It's useful
# for seeing the movement from different viewpoints (top-down, side, front).

def plot_marker_2d_projections(data, marker_id, color, xlim, ylim, zlim):
    """Plots the 2D projections (XY, XZ, YZ) for a single marker."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'2D Projections for Marker {marker_id.split("_")[0]}', fontsize=16, y=1.02)

    x_col, y_col, z_col = f'{marker_id}_X', f'{marker_id}_Y', f'{marker_id}_Z'

    if not all(col in data.columns for col in [x_col, y_col, z_col]):
        print(f"Skipping 2D projections for {marker_id}: Columns not found.")
        plt.close(fig) # Close the empty figure
        return

    # Plot XY Projection (Top-down view)
    ax = axes[0]
    ax.plot(data[x_col], data[y_col], label='Trajectory', color=color, alpha=0.8)
    ax.scatter(data[x_col].iloc[0], data[y_col].iloc[0], color=color, s=80, label='Start', edgecolor='black', marker='o')
    ax.scatter(data[x_col].iloc[-1], data[y_col].iloc[-1], color=color, s=80, label='End', edgecolor='black', marker='X')
    ax.set_title('X-Y Plane (Top View)')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box') # Maintain aspect ratio
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Plot XZ Projection (Side view)
    ax = axes[1]
    ax.plot(data[x_col], data[z_col], label='Trajectory', color=color, alpha=0.8)
    ax.scatter(data[x_col].iloc[0], data[z_col].iloc[0], color=color, s=80, label='Start', edgecolor='black', marker='o')
    ax.scatter(data[x_col].iloc[-1], data[z_col].iloc[-1], color=color, s=80, label='End', edgecolor='black', marker='X')
    ax.set_title('X-Z Plane (Side View)')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_aspect('equal', adjustable='box') # Maintain aspect ratio
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Plot YZ Projection (Front/Back view)
    ax = axes[2]
    ax.plot(data[y_col], data[z_col], label='Trajectory', color=color, alpha=0.8)
    ax.scatter(data[y_col].iloc[0], data[z_col].iloc[0], color=color, s=80, label='Start', edgecolor='black', marker='o')
    ax.scatter(data[y_col].iloc[-1], data[z_col].iloc[-1], color=color, s=80, label='End', edgecolor='black', marker='X')
    ax.set_title('Y-Z Plane (Front/Back View)')
    ax.set_xlabel('Y (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_xlim(ylim) # Note: use ylim for X axis here
    ax.set_ylim(zlim)
    ax.set_aspect('equal', adjustable='box') # Maintain aspect ratio
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to prevent title overlap
    plt.show()

# --- Generate Plots (Approach 2) ---
print("\nGenerating 2D projection plots for each marker...")
for pos in positions:
    plot_marker_2d_projections(drone_data, pos, colors[pos], (x_min, x_max), (y_min, y_max), (z_min, z_max))


# ============================================================================

# Approach 3: Time Series Plots for Each Coordinate (X, Y, Z)

# This shows how each coordinate (X, Y, Z) changes over time for all markers.
# Useful for seeing synchronization or specific movements along axes.

def plot_coordinate_time_series(data, axis, positions, colors, ylim):
    """Plots the time series for a specific coordinate (X, Y, or Z) for all markers."""
    plt.figure(figsize=(12, 6))

    for pos in positions:
        col_name = f'{pos}_{axis}'
        if col_name in data.columns:
            plt.plot(data['Time'], data[col_name], label=f'{pos} {axis}', color=colors[pos], alpha=0.8)
        else:
             print(f"Skipping {col_name} in time series plot: Column not found.")

    plt.title(f'{axis} Coordinate vs. Time for All Markers', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel(f'{axis} Position (meters)', fontsize=12)
    plt.ylim(ylim)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# --- Generate Plots (Approach 3) ---
print("\nGenerating time series plots for each coordinate...")
plot_coordinate_time_series(drone_data, 'X', positions, colors, (x_min, x_max))
plot_coordinate_time_series(drone_data, 'Y', positions, colors, (y_min, y_max))
plot_coordinate_time_series(drone_data, 'Z', positions, colors, (z_min, z_max))