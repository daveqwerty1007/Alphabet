import pandas as pd
import numpy as np
import folium

# General understing of Ripley’s K Function
# see the week3-4_RKFunction.pdf

# realization of Ripley’s K Function

def ripley_k_function(points, distances):
    n = len(points)
    area = (max(points[:, 0]) - min(points[:, 0])) * (max(points[:, 1]) - min(points[:, 1]))
    k_values = []

    for d in distances:
        count = 0
        for i in range(n):
            for j in range(n):
                if i != j and np.linalg.norm(points[i] - points[j]) <= d:
                    count += 1
        k_values.append((area / n**2) * count)
    
    return k_values


# dataset
# location information in the map of Santa Clara County.
# Sources: California Department Of Education (2021).
# https://data.sccgov.org/Government/Point-Of-Interest/asae-p5kt/about_data

file_path = 'Point_Of_Interest_20240616.csv'
df = pd.read_csv(file_path)

df = df.head(100)
#print(df.head())

# dataVis
points = df[['LATITUDE', 'LONGITUDE']].to_numpy()
points = points.astype(float)

distances = np.linspace(0, 5000, 50) #distance from 0 to 5km
k_values = ripley_k_function(points, distances)

m = folium.Map()

for idx, row in df.iterrows():
    folium.Marker([row['LATITUDE'], row['LONGITUDE']], popup=row['PLACENAME']).add_to(m)

m.save('santa_clara_map.html')

