import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Load data
df = pd.read_csv('combined_taxi_data.csv')

# Convert date_time to datetime object
df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')

# Remove invalid date_time entries
df = df.dropna(subset=['date_time'])

# Filter points within bounds
df = df[(df['longitude'] >= 116) & (df['longitude'] <= 116.8) &
        (df['latitude'] >= 39.3) & (df['latitude'] <= 40.5)]

# Load edge server locations
edge_servers = pd.read_csv('edge_server_locations.csv')

# Set font properties
font_prop = font_manager.FontProperties(family='Times New Roman', size=14)

# Create plot
plt.figure(figsize=(12, 10))
plt.scatter(df['longitude'], df['latitude'], s=5, alpha=0.3, label='All Taxi Points')
plt.scatter(edge_servers['longitude'], edge_servers['latitude'], c='red', s=70, marker='.', label='Edge Servers')

# Set labels and title with larger font
plt.xlabel('Longitude', fontproperties=font_prop)
plt.ylabel('Latitude', fontproperties=font_prop)
plt.title('Taxi Data Points and Edge Server Locations', fontproperties=font_prop)

# Set legend with larger font
plt.legend(prop=font_prop)

# Enable grid
plt.grid(True)

# Save high-quality PNG
plt.tight_layout()
plt.savefig('taxi_points_with_edge_servers.png', dpi=300, bbox_inches='tight')
plt.show()