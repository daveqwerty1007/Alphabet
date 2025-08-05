import numpy as np
import requests
import json
import itertools
import heapq
import matplotlib.pyplot as plt


overpass_url = "http://overpass-api.de/api/interpreter"

def query_osm(overpass_query):
    response = requests.get(overpass_url, params={'data': overpass_query})
    return response.json()

# malls
mall_query = """
[out:json];
area["ISO3166-2"="US-CA"];
(
  node["shop"="mall"](area);
  way["shop"="mall"](area);
  rel["shop"="mall"](area);
);
out center;
"""
data_mall = query_osm(mall_query)

# fitting rooms
fitting_query = """
[out:json];
area["ISO3166-2"="US-CA"];
(node["shop"="clothes"](area);
 way["shop"="clothes"](area);
 rel["shop"="clothes"](area);
);
out center;
"""
data_fitting = query_osm(fitting_query)

# libraries
library_query = """
[out:json];
area["ISO3166-2"="US-CA"];
(node["amenity"="hospital"](area);
 way["amenity"="hospital"](area);
 rel["amenity"="hospital"](area);
);
out center;
"""
data_library = query_osm(library_query)

# courts
basketball_query = """
[out:json];
area["ISO3166-2"="US-CA"];
(node["amenity"="school"]["name"~"High School"](area);
  way["amenity"="school"]["name"~"High School"](area);
  rel["amenity"="school"]["name"~"High School"](area);
  
  node["amenity"="college"](area);
  way["amenity"="college"](area);
  rel["amenity"="college"](area);

  node["amenity"="university"](area);
  way["amenity"="university"](area);
  rel["amenity"="university"](area);
);
out center;
"""
data_basketball = query_osm(basketball_query)

def extract_coordinates(json_data):
    elements = json_data['elements']
    coordinates = np.array([[element['lat'], element['lon']] for element in elements])
    return coordinates

def save_data(data, filename):
    np.save(filename, data)

fitting_coordinates = extract_coordinates(data_fitting)
save_data(fitting_coordinates, "data_fitting.npy")

mall_coordinates = extract_coordinates(data_mall)
save_data(mall_coordinates, "data_mall.npy")

basketball_coordinates = extract_coordinates(data_basketball)
save_data(basketball_coordinates, "data_basketball.npy")

library_coordinates = extract_coordinates(data_library)
save_data(library_coordinates, "data_library.npy")

data_basketball_1 = np.load("data_basketball.npy")
data_library_1 = np.load("data_library.npy")
data_mall_1 = np.load("data_mall.npy")
data_fitting_1 = np.load("data_fitting.npy")

data_combined = np.concatenate([data_basketball_1, data_library_1, data_mall_1, data_fitting_1], axis=0)

save_data(data_combined, "data_combined.npy")

def ripley_k(data, r):
    n = len(data)
    k = np.zeros(len(r))
    for i in range(n):
        for j, radius in enumerate(r):
            in_circle = np.linalg.norm(data[i] - data, axis=1) <= radius
            k[j] += np.sum(in_circle)
    return k / n

r = np.linspace(0, 35, 100)

k_basketball = ripley_k(data_basketball_1, r)
k_library = ripley_k(data_library_1, r)
k_mall = ripley_k(data_mall_1, r)
k_fitting = ripley_k(data_fitting_1, r)

k_csr = np.pi * r**2

# Calculate high and low boundaries for 95% confidence interval
#upper_bound = poisson.ppf(0.975, mu=k_csr)
#lower_bound = poisson.ppf(0.025, mu=k_csr)

min_lat, max_lat = 32.5, 42.0
min_lon, max_lon = -124.5, -114.0

num_lat_squares = 30
num_lon_squares = 60

lat_step = (max_lat - min_lat) / num_lat_squares
lon_step = (max_lon - min_lon) / num_lon_squares

k_matrix = np.zeros((num_lat_squares, num_lon_squares, 4))

def get_square(lat, lon, min_lat, min_lon, lat_step, lon_step):
    lat_idx = int((lat - min_lat) / lat_step)
    lon_idx = int((lon - min_lon) / lon_step)
    return lat_idx, lon_idx

def assign_k_values(data, k_values, k_index, k_matrix):
    for (lat, lon), k_value in zip(data, k_values):
        lat_idx, lon_idx = get_square(lat, lon, min_lat, min_lon, lat_step, lon_step)
        k_matrix[lat_idx, lon_idx, k_index] += k_value

assign_k_values(data_mall_1, k_mall, 0, k_matrix)
assign_k_values(data_library_1, k_library, 1, k_matrix)
assign_k_values(data_fitting_1, k_fitting, 2, k_matrix)
assign_k_values(data_basketball_1, k_basketball, 3, k_matrix)

def determine_threshold(k_matrix):
    k_values = k_matrix.flatten()
    mean_k = np.mean(k_values)
    std_k = np.std(k_values)
    threshold = mean_k + std_k
    return threshold

threshold = determine_threshold(k_matrix)

def heuristic(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

# A* Search 
def a_star_search(start, goals, k_matrix, min_lat, min_lon, lat_step, lon_step, threshold):
    open_set = []
    heapq.heappush(open_set, (0, start, [])) 
    
    visited = set()

    while open_set:
        current_cost, current_point, path = heapq.heappop(open_set)

        current_point_tuple = tuple(current_point)
        if current_point_tuple in visited:
            continue

        visited.add(current_point_tuple)
        path = path + [current_point_tuple]

        if len(path) == len(goals) + 1:  # Start + all goals
            return path, current_cost

        for goal in goals:
            goal_tuple = tuple(goal)
            if goal_tuple not in path:
                new_cost = current_cost +  heuristic(current_point, goal)
                
                lat_idx, lon_idx = get_square(goal[0], goal[1], min_lat, min_lon, lat_step, lon_step)
                if np.all(k_matrix[lat_idx, lon_idx, :] >= threshold):
                    estimated_cost = new_cost + heuristic(goal, goals[-1]) 
                    heapq.heappush(open_set, (estimated_cost, goal, path))

    return None, float('inf')


start_point = data_mall_1[0] 
goal_points = [data_library_1[0], data_fitting_1[0], data_basketball_1[0]]  


best_path, min_distance = a_star_search(start_point, goal_points, k_matrix, min_lat, min_lon, lat_step, lon_step, threshold)

print("Best Path:", best_path)
print("Minimum Distance:", min_distance)
