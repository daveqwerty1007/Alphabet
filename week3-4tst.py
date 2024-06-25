import pandas as pd
import numpy as np
import folium
import requests
import matplotlib.pyplot as plt

k_values_df = pd.read_csv('ripley_k_values.csv')

## fit point into data
plt.figure(figsize=(10, 6))
plt.plot(k_values_df['Distance'], k_values_df['K_Value'], label='K function for Drinking Water')

k_upper = k_values_df['K_Value'] + 0.01  
k_lower = k_values_df['K_Value'] - 0.01  
plt.fill_between(k_values_df['Distance'], k_lower, k_upper, color='gray', alpha=0.2, label='Confidence Interval')

plt.xlabel('Distance')
plt.ylabel('K function value')
plt.legend()
plt.title('Ripley’s K Function for Drinking Water in California')
plt.show()


# HTML map
overpass_url = "http://overpass-api.de/api/interpreter"
overpass_query = """
[out:json];
area["ISO3166-2"="US-CA"];
(
  node["amenity"="drinking_water"](area);
  way["amenity"="drinking_water"](area);
  rel["amenity"="drinking_water"](area);
);
out center;
"""
response = requests.get(overpass_url, params={'data': overpass_query})
data_water = response.json()

m = folium.Map(location=[37.6, -120.9], zoom_start=6)

for idx, element in enumerate(data_water['elements']):
    if 'lat' in element and 'lon' in element:
        lat = element['lat']
        lon = element['lon']
        name = element['tags'].get('name', 'No Name')
        folium.Marker(location=[lat, lon], popup=name, icon=folium.Icon(color='blue', icon='tint')).add_to(m)

# Save the map
m.save('drinking_water_map.html')
