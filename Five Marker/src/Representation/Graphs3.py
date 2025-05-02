import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
 
# Load data
file_path = 'Data/Processed/Modified_Drone_Data.csv'
df = pd.read_csv(file_path)

# Convert mm to meters
node_ids = ['1', '2', '3', '4', 'C']
for node in node_ids:
    for axis in ['X', 'Y', 'Z']:
        df[f'{node}_pos_{axis}'] /= 1000

# Function to get anomalies in 3D for each node
def detect_anomalies_3d(df, node):
    coords = df[[f'{node}_pos_X', f'{node}_pos_Y', f'{node}_pos_Z']]
    iso = IsolationForest(n_estimators=300, contamination=0.3, random_state=42)
    preds = iso.fit_predict(coords)
    anomalies = np.where(preds == -1)[0]
    return anomalies

# Create 3D scatter plot with anomalies
fig = go.Figure()

# Plot each node's motion
for node in node_ids:
    x = df[f'{node}_pos_X']
    y = df[f'{node}_pos_Y']
    z = df[f'{node}_pos_Z']
    anomalies = detect_anomalies_3d(df, node)

    # Plot motion path
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        name=f'Node {node}',
        marker=dict(size=3),
        line=dict(width=2),
        opacity=0.7
    ))

    # Plot anomalies
    fig.add_trace(go.Scatter3d(
        x=x.iloc[anomalies], y=y.iloc[anomalies], z=z.iloc[anomalies],
        mode='markers',
        name=f'Anomalies {node}',
        marker=dict(color='red', size=5, symbol='circle'),
        showlegend=True
    ))

# Update layout for interactivity
fig.update_layout(
    title="3D Drone Motion with Anomalies (All Nodes)",
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    legend=dict(x=0, y=1),
    height=800,
    template='plotly_white'
)

fig.show()