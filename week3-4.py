import pandas as pd
import numpy as np
import folium
import requests
import matplotlib.pyplot as plt

# General understing of Ripley’s K Function
# see the week3-4_RKFunction.pdf

# realization of Ripley’s K Function 
def ripley_k_function(points, distances):
    n = len(points)
    area = (max(points[:, 0]) - min(points[:, 0])) * (max(points[:, 1]) - min(points[:, 1]))
    k_values = []

    for d in distances:
        count = np.sum(np.linalg.norm(points[:, np.newaxis] - points, axis=2) <= d)
        k_values.append((area / n**2) * count)
    
    return k_values

"""
def ripley_k_function(points, distances):
    n = len(points)
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    area = (x_max - x_min) * (y_max - y_min) / 2 # southwest of cali is just water 
    
    dist_matrix = np.sqrt(((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2).sum(axis=2))

    k_values = np.zeros_like(distances)

    for idx, d in enumerate(distances):
        count = np.sum(dist_matrix <= d) - n 
        k_values[idx] = (area / (n**2)) * count
    
    return k_values
"""

# dataset
# location information in the map of Santa Clara County.
# Sources: California Department Of Education (2021).
# https://data.sccgov.org/Government/Point-Of-Interest/asae-p5kt/about_data

"""
file_path = 'Point_Of_Interest_20240616.csv'
df = pd.read_csv(file_path)

df = df.head(100)
print(df.head())
"""

# Overpass API
overpass_url = "http://overpass-api.de/api/interpreter"
overpass_query = """
[out:json][timeout:25];
area["name"="California"];
node(area)["amenity"="drinking_water"];
out geom;
"""
response = requests.get(overpass_url, params={'data': overpass_query})
data_water = response.json()

#points = df[['LATITUDE', 'LONGITUDE']].to_numpy()
#points = points.astype(float)

points = []
for element in data_water['elements']:
    if 'lat' in element and 'lon' in element:
        points.append([element['lat'], element['lon']])
points = np.array(points)

distances = np.linspace(0, 5000, 50) #distance from 0 to 5km
k_values = ripley_k_function(points, distances)

k_df = pd.DataFrame({'Distance': distances, 'K_Value': k_values})
k_df.to_csv('ripley_k_values.csv', index=False)

