import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Load the data
file_path = 'Data/Processed/Modified_Drone_Data.csv'
df = pd.read_csv(file_path)

# Convert position units from mm to meters
node_ids = ['1', '2', '3', '4', 'C']
for node in node_ids:
    for axis in ['X', 'Y', 'Z']:
        df[f'{node}_pos_{axis}'] /= 1000

# Time array
time = df['Time']

# Calculate velocity and acceleration
for node in node_ids:
    for axis in ['X', 'Y', 'Z']:
        pos = df[f'{node}_pos_{axis}']
        df[f'{node}_v{axis.lower()}'] = np.gradient(pos, time)
        df[f'{node}_a{axis.lower()}'] = np.gradient(df[f'{node}_v{axis.lower()}'], time)

# Function to plot velocity or acceleration anomalies in 3 subplots (X, Y, Z)
def plot_node_motion_anomalies(df, node, quantity='velocity'):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Node {node} - {quantity.title()} with Anomalies (X, Y, Z)', fontsize=16)

    for i, axis in enumerate(['x', 'y', 'z']):
        if quantity == 'velocity':
            col = f'{node}_v{axis}'
            ylabel = f'Velocity {axis.upper()} (m/s)'
        else:
            col = f'{node}_a{axis}'
            ylabel = f'Acceleration {axis.upper()} (m/s²)'

        values = df[[col]]
        iso = IsolationForest(n_estimators=300, contamination='auto', random_state=42)
        preds = iso.fit_predict(values)
        anomalies = np.where(preds == -1)[0]

        # Plot
        axes[i].plot(time, df[col], label='Data', color='blue')
        axes[i].scatter(time.iloc[anomalies], df[col].iloc[anomalies],
                        color='red', label='Anomalies', s=50, edgecolor='black')
        axes[i].set_ylabel(ylabel)
        axes[i].legend()
        axes[i].grid(True, linestyle='--')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Run for all nodes
for node in node_ids:
    plot_node_motion_anomalies(df, node, quantity='acceleration')
    plot_node_motion_anomalies(df, node, quantity='velocity')
